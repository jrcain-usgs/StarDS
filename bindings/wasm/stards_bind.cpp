#include "stards.h"

#include <string>
#include <vector>
#include <utility>
#include <limits>
#include <map>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <variant>

#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/em_js.h>
#include <emscripten/val.h>

using namespace emscripten;
using star::StarDataset;
using star::DataType;
using star::CompressionAlgorithm;
using star::LayerView;
using star::MetadataValue;
using star::NDArray;
using star::Slice;
using star::StarConfig;
using star::OpenOptions;
using star::ValueVariant;

namespace {

/**
 * @brief Copies NDArray into JS typed array.
 * 
 * @param arr An NDArray
 * @param js_ctor JS ArrayType (Int32Array, Float64Array, ...)
 * 
 * @return A JS Typed Array
 */
template <typename T>
val to_typed_array(const NDArray<T>& arr, const char* js_ctor) {
    const std::vector<T>& d = arr.data();
    val view(typed_memory_view(d.size(), d.data()));
    return val::global(js_ctor).new_(view);
}


/**
 * @brief Translates C++ DataType to JS Type and calls fn(), passing in the type.
 * 
 * @param dt A C++ DataType (ie, INT8)
 * @param ctx Name of the calling function (for error logging)
 * @param fn Callable fn(Type{}, "TypedArrName"). fn can get Type via decltype(arg1).
 * 
 * @return Returns what fn returns
 */
template <typename F>
auto dispatch_numeric(DataType dt, const char* ctx, F&& fn) -> decltype(fn(int8_t{}, "")) {
    switch (dt) {
        case DataType::INT8:    return fn(int8_t{},   "Int8Array");
        case DataType::INT16:   return fn(int16_t{},  "Int16Array");
        case DataType::INT32:   return fn(int32_t{},  "Int32Array");
        case DataType::UINT8:   return fn(uint8_t{},  "Uint8Array");
        case DataType::UINT16:  return fn(uint16_t{}, "Uint16Array");
        case DataType::UINT32:  return fn(uint32_t{}, "Uint32Array");
        case DataType::FLOAT32: return fn(float{},    "Float32Array");
        case DataType::FLOAT64: return fn(double{},   "Float64Array");
        // 64-bit ints exceed JS Number safe range; expose via BigInt arrays.
        case DataType::INT64:   return fn(int64_t{},  "BigInt64Array");
        case DataType::UINT64:  return fn(uint64_t{}, "BigUint64Array");
        default:
            throw std::runtime_error(std::string(ctx) + "(): unsupported dtype");
    }
}


// Translates string to C++ DataType
DataType datatype_from_string(const std::string& s) {
    if (s == "int8")    return DataType::INT8;
    if (s == "int16")   return DataType::INT16;
    if (s == "int32")   return DataType::INT32;
    if (s == "int64")   return DataType::INT64;
    if (s == "uint8")   return DataType::UINT8;
    if (s == "uint16")  return DataType::UINT16;
    if (s == "uint32")  return DataType::UINT32;
    if (s == "uint64")  return DataType::UINT64;
    if (s == "float32") return DataType::FLOAT32;
    if (s == "float64") return DataType::FLOAT64;
    if (s == "string")  return DataType::STRING;
    throw std::runtime_error("unknown dtype: " + s);
}

/**
 * @brief Parses OpenOptions from a JS object.
 * 
 * @param opts Recognized options: { layerInheritance: bool, prefetchWholeBelowBytes: number }.
*/
OpenOptions parse_open_options(const val& opts) {
    OpenOptions o;  // defaults (layer_inheritance=false, prefetch_whole_below_bytes=8MiB)
    if (opts.isUndefined() || opts.isNull()) return o;
    const val li = opts["layerInheritance"];
    if (!li.isUndefined()) o.layer_inheritance = li.as<bool>();
    const val pw = opts["prefetchWholeBelowBytes"];
    if (!pw.isUndefined()) o.prefetch_whole_below_bytes = static_cast<size_t>(pw.as<double>());
    return o;
}

/**
 * @brief Checks if array (as opposed to bare number/string)
 * 
 * @return true for JS Array or TypedArray (anything with numeric .length)
 */
bool is_array_like(const val& v) {
    if (v.isString() || v.isNull() || v.isUndefined()) return false;
    return v["length"].isNumber();
}

/**
 * @brief Reads a JS value as C++ Type.
 * 
 * @return value cast to double (except 64-bit int -> BigInt)
 */
template <typename T>
T js_to_scalar(const val& v) {
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        return v.as<T>();
    } else {
        return static_cast<T>(v.as<double>());
    }
}


/**
 * @brief Converts JS single-value or array-like to 1-D NDArray
 * 
 * @param v array, or single-value
 */
template <typename T>
NDArray<T> js_to_ndarray(const val& v) {
    if (is_array_like(v)) {
        const unsigned n = v["length"].as<unsigned>();
        std::vector<T> data;
        data.reserve(n);
        for (unsigned i = 0; i < n; ++i) data.push_back(js_to_scalar<T>(v[i]));
        return NDArray<T>(std::move(data), std::vector<size_t>{n});
    }
    return NDArray<T>(std::vector<T>{js_to_scalar<T>(v)}, std::vector<size_t>{});
}


/**
 * @brief Converts JS string or array of strings to 1-D NDArray
 * 
 * @param v string, or array of strings
 */
NDArray<std::string> js_to_string_ndarray(const val& v) {
    if (is_array_like(v)) {
        const unsigned n = v["length"].as<unsigned>();
        std::vector<std::string> data;
        data.reserve(n);
        for (unsigned i = 0; i < n; ++i) data.push_back(v[i].as<std::string>());
        return NDArray<std::string>(std::move(data), std::vector<size_t>{n});
    }
    return NDArray<std::string>(std::vector<std::string>{v.as<std::string>()},
                                std::vector<size_t>{});
}

// Copy JS number array (Array or TypedArray) into std::vector<T>.
// Mostly a catch for BigInt
template <typename T>
std::vector<T> js_numbers_to_vector(const val& v) {
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        const unsigned n = v["length"].as<unsigned>();
        std::vector<T> out;
        out.reserve(n);
        for (unsigned i = 0; i < n; ++i) out.push_back(v[i].as<T>());
        return out;
    } else {
        return convertJSArrayToNumberVector<T>(v);
    }
}


/**
 * @brief Allocates a vector based on shape (or size).
 * 
 * Defaults to double for JS numbers.
 * 
 * @param shape_arg Shape (ie, [10, 20] for a 10x20 2D Array)
 * @param total Size to allocate (fallback if shape not provided)
 */
std::vector<size_t> parse_shape(const val& shape_arg, size_t total) {
    if (is_array_like(shape_arg)) {
        const unsigned nd = shape_arg["length"].as<unsigned>();
        std::vector<size_t> shape;
        shape.reserve(nd);
        for (unsigned i = 0; i < nd; ++i)
            shape.push_back(static_cast<size_t>(shape_arg[i].as<double>()));
        return shape;
    }
    return {total};
}

// Build an NDArray<T> for a put(): bulk-copy the JS values, then apply the shape.
template <typename T>
NDArray<T> build_ndarray(const val& value, const val& shape_arg) {
    std::vector<T> data = js_numbers_to_vector<T>(value);
    std::vector<size_t> shape = parse_shape(shape_arg, data.size());
    return NDArray<T>(std::move(data), shape);
}

// Build a String NDArray. 
// Accepts single string (for a length 1 array), or an array of strings.
NDArray<std::string> build_string_ndarray(const val& value, const val& shape_arg) {
    std::vector<std::string> data;
    if (value.isString()) {
        data.push_back(value.as<std::string>());
    } else {
        const unsigned n = value["length"].as<unsigned>();
        data.reserve(n);
        for (unsigned i = 0; i < n; ++i) data.push_back(value[i].as<std::string>());
    }
    std::vector<size_t> shape;
    if (is_array_like(shape_arg)) shape = parse_shape(shape_arg, data.size());
    else if (!value.isString()) shape = {data.size()};  // else: scalar string
    return NDArray<std::string>(std::move(data), shape);
}


// Parse 1D slice from array [start, stop, step?].
// Defaults to start= 0, stop=dim, step=1
Slice parse_one_slice(const val& startStopStep, size_t dim) {
    double start = 0, stop = static_cast<double>(dim), step = 1;
    if (is_array_like(startStopStep)) {
        const unsigned n = startStopStep["length"].as<unsigned>();
        if (n > 0) start = startStopStep[0].as<double>();
        if (n > 1) stop = startStopStep[1].as<double>();
        if (n > 2) step = startStopStep[2].as<double>();
    }
    const size_t s = start <= 0 ? 0 : std::min(static_cast<size_t>(start), dim);
    size_t e = stop <= 0 ? 0 : std::min(static_cast<size_t>(stop), dim);
    if (e < s) e = s;
    const size_t st = step < 1 ? 1 : static_cast<size_t>(step);
    return Slice{s, e, st};
}


// 
// Example sliceWindows 
/**
 * @brief Parse JS window array into vector of Slices.
 * 
 * @param shape Used as bound/error check for slices
 * @param sliceWindows ie, for 2D slice: [[start, stop, step?], [start, stop, step?]]
 * 
 * @return vector of Slice spec
 */
std::vector<Slice> parse_slices(const std::vector<size_t>& shape, const val& sliceWindows) {
    const unsigned nd = shape.size();
    const unsigned given =
        is_array_like(sliceWindows) ? sliceWindows["length"].as<unsigned>() : 0;
    if (given > nd) {
        throw std::runtime_error("getSliceND: more slice windows (" + std::to_string(given) +
                                 ") than array dimensions (" + std::to_string(nd) + ")");
    }
    std::vector<Slice> out;
    out.reserve(nd);
    for (unsigned i = 0; i < nd; ++i) {
        out.push_back(i < given ? parse_one_slice(sliceWindows[i], shape[i])
                                : star::slice_all(shape[i]));
    }
    return out;
}


// Returns metadata value in JS type.
// If single element, returns scalar or str. Otherwise, array.
val meta_value_to_js(const MetadataValue& mv) {
    if (mv.dtype == DataType::STRING) {
        NDArray<std::string> a = mv.as<std::string>();
        if (mv.is_scalar()) return a.size() ? val(a.flat(0)) : val(std::string());
        val out = val::array();
        for (size_t i = 0; i < a.size(); ++i) out.set(i, val(a.flat(i)));
        return out;
    }
    const bool scalar = mv.is_scalar();
    return dispatch_numeric(mv.dtype, "metaGet", [&](auto tag, const char* ctor) {
        using T = decltype(tag);
        NDArray<T> a = mv.as<T>();
        if constexpr (!std::is_same_v<T, int64_t> && !std::is_same_v<T, uint64_t>) {
            if (scalar) return val(a.flat(0));
        }
        return to_typed_array(a, ctor);
    });
}

// ---- NDArray (first-class, dtype-erased) ---------------------------------

// Read an index/shape argument (JS number array) into size_t dims.
std::vector<size_t> to_size_vector(const val& a) {
    std::vector<size_t> out;
    if (is_array_like(a)) {
        const unsigned n = a["length"].as<unsigned>();
        out.reserve(n);
        for (unsigned i = 0; i < n; ++i) out.push_back(static_cast<size_t>(a[i].as<double>()));
    }
    return out;
}

// DataType of the NDArray currently held in a ValueVariant.
DataType variant_dtype(const ValueVariant& var) {
    return std::visit([](const auto& arr) {
        return star::TypeToDataType<typename std::decay_t<decltype(arr)>::value_type>::value;
    }, var);
}

// Build a ValueVariant (an NDArray<T>) from JS (value, shape, dtype)
ValueVariant build_variant(const val& value, const val& shape, const std::string& dtype) {
    if (dtype == "string") return build_string_ndarray(value, shape);
    return dispatch_numeric(datatype_from_string(dtype), "NDArray",
        [&](auto tag, const char*) -> ValueVariant {
            return build_ndarray<decltype(tag)>(value, shape);
        });
}

// Read array `key` from a dataset into a ValueVariant
ValueVariant read_variant(StarDataset& ds, const std::string& key) {
    const DataType dt = ds.dtype_of(key);
    if (dt == DataType::STRING) return ds.get<std::string>(key);
    return dispatch_numeric(dt, "getArray", [&](auto tag, const char*) -> ValueVariant {
        return ds.get<decltype(tag)>(key);
    });
}

// Flat JS view of a variant's data: a typed array (numeric) or Array<string>.
val variant_to_js_data(const ValueVariant& var, DataType dt) {
    if (dt == DataType::STRING) {
        const NDArray<std::string>& arr = std::get<NDArray<std::string>>(var);
        val out = val::array();
        for (size_t i = 0; i < arr.size(); ++i) out.set(i, val(arr.flat(i)));
        return out;
    }
    return dispatch_numeric(dt, "NDArray.data", [&](auto tag, const char* ctor) {
        using T = decltype(tag);
        return to_typed_array(std::get<NDArray<T>>(var), ctor);
    });
}


class JsNDArray {
public:
    // new Module.NDArray(value, shape, dtype) — same (value, shape, dtype) convention
    // as Dataset.put (value: TypedArray/Array/string; shape: array-like, [] = scalar,
    // or null = 1-D of the data length).
    JsNDArray(val value, val shape, const std::string& dtype)
        : m_var(build_variant(value, shape, dtype)), m_dtype(variant_dtype(m_var)) {}

    // Wrap an existing variant (used by getArray/factories); not a JS constructor.
    explicit JsNDArray(ValueVariant var) : m_var(std::move(var)), m_dtype(variant_dtype(m_var)) {}

    std::string dtype() const { return std::string(star::datatype_to_string(m_dtype)); }

    val shape() const {
        return std::visit([](const auto& arr) {
            val out = val::array();
            const auto& s = arr.shape();
            for (size_t i = 0; i < s.size(); ++i) out.set(i, val(static_cast<double>(s[i])));
            return out;
        }, m_var);
    }

    double size() const {
        return std::visit([](const auto& arr) { return static_cast<double>(arr.size()); }, m_var);
    }
    double ndim() const {
        return std::visit([](const auto& arr) { return static_cast<double>(arr.dimension()); }, m_var);
    }

    // Flat data (row-major) as a typed array (numeric) or Array<string>.
    val data() const { return variant_to_js_data(m_var, m_dtype); }

    // Single element at `indices` (a JS array, one entry per dim). Numeric comes back
    // as Number (BigInt for 64-bit ints); strings as string.
    val at(val indices) const {
        std::vector<size_t> idx = to_size_vector(indices);
        return std::visit([&](const auto& arr) -> val {
            using T = typename std::decay_t<decltype(arr)>::value_type;
            const T& v = arr.at(idx);
            if constexpr (std::is_same_v<T, std::string>) return val(v);
            else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) return val(v);
            else return val(static_cast<double>(v));
        }, m_var);
    }

    // Reshape in place; total element count must match (else a catchable throw).
    void reshape(val new_shape) {
        std::vector<size_t> s = to_size_vector(new_shape);
        std::visit([&](auto& arr) { arr.reshape(s); }, m_var);
    }

    // Factories (numeric dtypes only; a string dtype throws "unsupported dtype").
    static JsNDArray zeros(val shape, const std::string& dtype) {
        std::vector<size_t> s = to_size_vector(shape);
        return dispatch_numeric(datatype_from_string(dtype), "NDArray.zeros",
            [&](auto tag, const char*) -> JsNDArray {
                return JsNDArray(ValueVariant(NDArray<decltype(tag)>::zeros(s)));
            });
    }
    static JsNDArray ones(val shape, const std::string& dtype) {
        std::vector<size_t> s = to_size_vector(shape);
        return dispatch_numeric(datatype_from_string(dtype), "NDArray.ones",
            [&](auto tag, const char*) -> JsNDArray {
                return JsNDArray(ValueVariant(NDArray<decltype(tag)>::ones(s)));
            });
    }
    static JsNDArray full(val shape, val value, const std::string& dtype) {
        std::vector<size_t> s = to_size_vector(shape);
        return dispatch_numeric(datatype_from_string(dtype), "NDArray.full",
            [&](auto tag, const char*) -> JsNDArray {
                using T = decltype(tag);
                return JsNDArray(ValueVariant(NDArray<T>::full(s, js_to_scalar<T>(value))));
            });
    }

    // Move the held variant out (for putArray) — leaves this handle empty.
    ValueVariant take() { return std::move(m_var); }
    // Borrow the held variant (for metaPutArray, which copies it in).
    const ValueVariant& variant() const { return m_var; }

private:
    ValueVariant m_var;
    DataType m_dtype;
};

// JS handle over a dataset layer (C++ LayerView). Holds a shared_ptr so it keeps
// the base dataset alive independently of the Dataset handle it came from. get()/
// put() mirror Dataset's, scoped to this layer (with base-layer fallback when the
// dataset has inheritance enabled); the meta* methods mirror the dataset metadata
// methods, scoped to this layer.
class JsLayer {
public:
    explicit JsLayer(std::shared_ptr<LayerView> layer) : m_layer(std::move(layer)) {}

    std::string name() const { return m_layer->name(); }

    // Keys in this layer, as Array<string>. NOTE: reflects LayerView::keys(), which
    // (per the TODO in stards.h) reports layer metadata + inherited keys but NOT
    // layer-local ARRAY keys written via put() — those are stored under a prefixed
    // key the presence check misses. Use get()/metaKeys(); don't rely on keys()/has()
    // to enumerate arrays you put() into a layer.
    val keys() const {
        std::vector<std::string> k = m_layer->keys();
        val out = val::array();
        for (size_t i = 0; i < k.size(); ++i) out.set(i, val(k[i]));
        return out;
    }

    // Faithful to LayerView::contains — see the keys() note: returns false for a
    // layer-local array key even though get() would return it.
    bool contains(const std::string& key) const { return m_layer->contains(key); }

    // Read array `key` from this layer. Numeric -> typed array; STRING -> Array<string>.
    val get(const std::string& key) const {
        const DataType dt = layer_dtype(key);
        if (dt == DataType::STRING) {
            NDArray<std::string> a = m_layer->get<std::string>(key);
            val out = val::array();
            for (size_t i = 0; i < a.size(); ++i) out.set(i, val(a.flat(i)));
            return out;
        }
        return dispatch_numeric(dt, "layer.get", [&](auto tag, const char* ctor) {
            return to_typed_array(m_layer->get<decltype(tag)>(key), ctor);
        });
    }

    // Write array `key` into this layer. Same (value, shape, dtype) convention as
    // Dataset.put; requires the dataset opened writable.
    void put(const std::string& key, val value, val shape, const std::string& dtype) {
        if (dtype == "string") {
            m_layer->put(key, build_string_ndarray(value, shape));
            return;
        }
        dispatch_numeric(datatype_from_string(dtype), "layer.put", [&](auto tag, const char*) {
            m_layer->put(key, build_ndarray<decltype(tag)>(value, shape));
            return val::undefined();
        });
    }

    // Read array `key` as a first-class NDArray (shape-aware handle, must .delete()).
    JsNDArray get_array(const std::string& key) const {
        const DataType dt = layer_dtype(key);
        if (dt == DataType::STRING) return JsNDArray(ValueVariant(m_layer->get<std::string>(key)));
        return JsNDArray(dispatch_numeric(dt, "layer.getArray",
            [&](auto tag, const char*) -> ValueVariant { return m_layer->get<decltype(tag)>(key); }));
    }

    // Write an NDArray into this layer (moves its data across — no extra copy).
    void put_array(const std::string& key, JsNDArray arr) {
        std::visit([&](auto&& a) { m_layer->put(key, std::move(a)); }, arr.take());
    }

    // Layer-scoped metadata — same shapes as the Dataset meta* methods.
    val meta_keys() const {
        std::vector<std::string> k = m_layer->meta.keys();
        val out = val::array();
        for (size_t i = 0; i < k.size(); ++i) out.set(i, val(k[i]));
        return out;
    }
    bool meta_contains(const std::string& key) const { return m_layer->meta.contains(key); }
    val meta_get(const std::string& key) const {
        std::shared_ptr<MetadataValue> mv = m_layer->meta.get(key);
        if (!mv) return val::null();
        return meta_value_to_js(*mv);
    }
    void meta_put(const std::string& key, val value, const std::string& dtype) {
        if (dtype == "string") {
            m_layer->meta.put(key, js_to_string_ndarray(value));
            return;
        }
        dispatch_numeric(datatype_from_string(dtype), "layer.metaPut", [&](auto tag, const char*) {
            m_layer->meta.put(key, js_to_ndarray<decltype(tag)>(value));
            return val::undefined();
        });
    }
    // Store an NDArray as layer metadata (any shape). meta.put copies it in, so the
    // caller's NDArray stays valid. The loader routes metaPut(key, ndarray) here.
    void meta_put_array(const std::string& key, const JsNDArray& arr) {
        std::visit([&](const auto& a) { m_layer->meta.put(key, a); }, arr.variant());
    }
    void meta_remove(const std::string& key) { m_layer->meta.remove(key); }

private:
    // Resolve the dtype of a layer array key. LayerView exposes no dtype accessor,
    // so we mirror its internal storage-key scheme (see LayerView::get) and ask the
    // base dataset: the layer-prefixed key if present, else the unprefixed base key
    // when the dataset has inheritance enabled.
    DataType layer_dtype(const std::string& key) const {
        std::shared_ptr<StarDataset> base = m_layer->base();
        const std::string& lname = m_layer->name();
        const std::string storage_key =
            lname == "__base__" ? key : ("__layer_" + lname + "__:" + key);
        if (base->contains(storage_key)) return base->dtype_of(storage_key);
        if (lname != "__base__" && base->layer_inheritance() && base->contains(key)) {
            return base->dtype_of(key);
        }
        throw std::runtime_error("Key not found in layer: " + key);
    }

    std::shared_ptr<LayerView> m_layer;
};

// A thin JS-facing handle around a StarDataset shared_ptr.
class JsDataset {
public:
    // Opens read-only. Under WASM a URL routes through the fetch() backend; a
    // bare path is a (virtual) local file. Throws on failure -> JS exception.
    explicit JsDataset(const std::string& path)
        : m_ds(StarDataset::open(path, "r")) {}

    // Opens with an explicit mode ("r", "w"/"rw"/"a"). Writable modes are needed
    // for the metadata writers (metaPut/metaRemove/metaClear) and only make sense
    // on a local/virtual path — a remote URL is read-only over fetch().
    JsDataset(const std::string& path, const std::string& mode)
        : m_ds(StarDataset::open(path, mode)) {}

    // Opens with a mode plus read-time OpenOptions from a JS object, e.g.
    // new Dataset(url, "r", { layerInheritance: true, prefetchWholeBelowBytes: 0 }).
    // (layerInheritance can also be toggled after open via setLayerInheritance.)
    JsDataset(const std::string& path, const std::string& mode, val opts)
        : m_ds(StarDataset::open(path, mode, parse_open_options(opts))) {}

    // Wrap an already-constructed dataset (used by the create() factory below).
    // Not registered as a JS constructor.
    explicit JsDataset(std::shared_ptr<StarDataset> ds) : m_ds(std::move(ds)) {}

    // Array keys present in the dataset (returned to JS as an Array<string>).
    val keys() const {
        std::vector<std::string> k = m_ds->get_all_keys();
        val arr = val::array();
        for (size_t i = 0; i < k.size(); ++i) arr.set(i, val(k[i]));
        return arr;
    }

    std::string dtype(const std::string& key) const {
        return std::string(star::datatype_to_string(m_ds->dtype_of(key)));
    }

    val shape(const std::string& key) const {
        // Metadata-only: shape_of() reads dims from the index, no data download.
        std::vector<size_t> s = m_ds->shape_of(key);
        val out = val::array();
        for (size_t i = 0; i < s.size(); ++i) out.set(i, val(static_cast<double>(s[i])));
        return out;
    }

    // Return the whole array for `key` as a JS typed array
    val get(const std::string& key) const {
        const DataType dt = m_ds->dtype_of(key);
        if (dt == DataType::STRING) {
            NDArray<std::string> a = m_ds->get<std::string>(key);
            val out = val::array();
            for (size_t i = 0; i < a.size(); ++i) out.set(i, val(a.flat(i)));
            return out;
        }
        return dispatch_numeric(dt, "get", [&](auto tag, const char* ctor) {
            return to_typed_array(m_ds->get<decltype(tag)>(key), ctor);
        });
    }

    // Write an array entry. `value` is a TypedArray/Array of numbers, or a
    // string/Array<string> when dtype is "string"; `dtype` names the on-disk
    // element type (see datatype_from_string); `shape` gives the dims (an
    // array-like, or omit/[] — [] is a scalar, omitted is 1-D of the data length).
    // put() itself never checks the open mode — the entry is held in memory and
    // overwriting an existing key is allowed — so it works even on a read-only
    // handle. Persisting is where the mode matters: flush() throws in read-only
    // mode, while writeBytes()/saveTo() serialize any dataset (see below).
    // Takes a FLAT value + explicit shape/dtype. The loader (stards.mjs) adds the
    // conveniences on top: inferring dtype from the value, defaulting shape to 1-D,
    // and flattening a nested array (e.g. [[1,2,3],[4,5,6]]) into flat data + a
    // derived shape before it reaches here.
    void put(const std::string& key, val value, val shape, const std::string& dtype) {
        if (dtype == "string") {
            m_ds->put(key, build_string_ndarray(value, shape));
            return;
        }
        dispatch_numeric(datatype_from_string(dtype), "put", [&](auto tag, const char*) {
            m_ds->put(key, build_ndarray<decltype(tag)>(value, shape));
            return val::undefined();
        });
    }

    // Read array `key` as a first-class NDArray (a shape-aware handle you must
    // .delete()). Complements get(), which returns a bare flat typed array.
    JsNDArray get_array(const std::string& key) const { return JsNDArray(read_variant(*m_ds, key)); }

    // Write an NDArray under `key` (moves its data across — no extra copy). The
    // counterpart of getArray(); equivalent to put() with the array's dtype/shape.
    void put_array(const std::string& key, JsNDArray arr) {
        std::visit([&](auto&& a) { m_ds->put(key, std::move(a)); }, arr.take());
    }

    // Persist pending writes to the dataset's backing path. Under WASM that path is
    // in the virtual filesystem. In browser, prefer writeBytes() to get the image back as bytes.
    void flush() { m_ds->flush(); }

    // Serialize the dataset to a .stards image and return it as a
    // Uint8Array. Just returns bytes, doesn't touch the source file.
    val write_bytes() {
        std::vector<char> bytes = m_ds->write_bytes();
        val view(typed_memory_view(bytes.size(),
                                   reinterpret_cast<const uint8_t*>(bytes.data())));
        return val::global("Uint8Array").new_(view);
    }

    // Persist the dataset to `path` in the virtual filesystem.
    // Makes a second copy, like "Save As..." in many apps.
    void save_to(const std::string& path) { m_ds->save_to(path); }

    // Best-effort flush + release, mirroring the destructor: a no-op (not an error)
    // for a read-only dataset. The JS handle itself must still be .delete()'d.
    void close() { m_ds->close(); }

    // Read a string-valued entry (header/attribute style) as a plain JS string,
    // from the metadata block or from a 1-element string column. Returns "" if the
    // key is absent, so callers can fall back to a default without a try/catch.
    std::string meta_string(const std::string& key) const {
        try {
            if (m_ds->meta.contains(key)) {
                NDArray<std::string> arr = m_ds->meta.get(key)->as<std::string>();
                return arr.size() ? arr.flat(0) : std::string();
            }
            if (m_ds->contains(key)) {
                NDArray<std::string> arr = m_ds->get<std::string>(key);
                return arr.size() ? arr.flat(0) : std::string();
            }
        } catch (const std::exception&) {
            // Wrong kind of entry for this key — treat as absent.
        }
        return std::string();
    }

    // Read a metadata-block entry `key` and return it as its natural JS type: a
    // number/string for scalars, a typed array for arrays, or null if the key is
    // absent (meta.get() yields nullptr for a miss, so we check before deref'ing).
    //
    // This is the general form of metaString(): it dispatches on dtype instead of
    // assuming string, so numeric attributes come back as numbers/typed arrays.
    val meta_get(const std::string& key) const {
        std::shared_ptr<MetadataValue> mv = m_ds->meta.get(key);
        if (!mv) return val::null();
        return meta_value_to_js(*mv);
    }

    // Names of all metadata-block entries (Array<string>). Cheap: reads the
    // registries, decodes no values.
    val meta_keys() const {
        std::vector<std::string> k = m_ds->get_metadata_keys();
        val out = val::array();
        for (size_t i = 0; i < k.size(); ++i) out.set(i, val(k[i]));
        return out;
    }

    bool meta_contains(const std::string& key) const { return m_ds->meta.contains(key); }

    // dtype name of a metadata entry ("int32", "float64", "string", ...), or "" if
    // the key is absent.
    std::string meta_dtype(const std::string& key) const {
        std::shared_ptr<MetadataValue> mv = m_ds->meta.get(key);
        return mv ? mv->type_name() : std::string();
    }

    // Shape of a metadata entry as Array<number> (empty for a scalar, [] if absent).
    val meta_shape(const std::string& key) const {
        val out = val::array();
        std::shared_ptr<MetadataValue> mv = m_ds->meta.get(key);
        if (mv) {
            for (size_t i = 0; i < mv->shape.size(); ++i)
                out.set(i, val(static_cast<double>(mv->shape[i])));
        }
        return out;
    }

    // All metadata as a plain JS object { key: naturalValue }.
    val meta_get_all() const {
        std::map<std::string, MetadataValue> all = m_ds->meta.get_all();
        val obj = val::object();
        for (const auto& [k, mv] : all) obj.set(k, meta_value_to_js(mv));
        return obj;
    }

    // Write a metadata entry. `dtype` picks the on-disk element type (see
    // datatype_from_string); `value` may be a scalar (-> scalar entry) or an
    // array-like (-> 1-D). Requires the dataset opened writable, else meta.put
    // throws. Overwrites any existing entry for `key`.
    void meta_put(const std::string& key, val value, const std::string& dtype) {
        if (dtype == "string") {
            m_ds->meta.put(key, js_to_string_ndarray(value));
            return;
        }
        dispatch_numeric(datatype_from_string(dtype), "metaPut", [&](auto tag, const char*) {
            m_ds->meta.put(key, js_to_ndarray<decltype(tag)>(value));
            return val::undefined();
        });
    }
    // Store an NDArray as metadata (any shape). meta.put copies it in, so the
    // caller's NDArray stays valid. The loader routes metaPut(key, ndarray) here.
    void meta_put_array(const std::string& key, const JsNDArray& arr) {
        std::visit([&](const auto& a) { m_ds->meta.put(key, a); }, arr.variant());
    }

    void meta_remove(const std::string& key) { m_ds->meta.remove(key); }
    void meta_clear() { m_ds->meta.clear(); }

    // True if `key` is stored as blocks and can be windowed with getSlice().
    // Metadata-block arrays are whole-array only (see StarDataset::is_sliceable).
    bool is_sliceable(const std::string& key) const { return m_ds->is_sliceable(key); }

    // Return elements [start, start+count) of the 1-D array `key` as a typed array.
    //
    // The point of this over get(): a slice reads only the compressed blocks that
    // cover the window, so a caller streaming a large column pays for the bytes it
    // actually wants instead of downloading the whole array up front. Streaming
    // consumers (the docs-site hero, for one) live on this.
    val get_slice(const std::string& key, double start, double count) const {
        const std::vector<Slice> s = {slice_1d(key, start, count)};
        return dispatch_numeric(m_ds->dtype_of(key), "getSlice", [&](auto tag, const char* ctor) {
            return to_typed_array(m_ds->get_slice<decltype(tag)>(key, s), ctor);
        });
    }

    // N-dimensional strided slice (1-D/2-D/3-D — the ranks the store slices). `windows`
    // is an array of per-dimension {start,stop,step?} / [start,stop,step?] specs (see
    // parse_slices: windows clamp, fields default, and omitted trailing dims are taken
    // in full). Like getSlice, only the covering compressed blocks are read.
    //
    // Returns an OBJECT { data, shape }: `data` is the flat (row-major) typed array, 
    // `shape` its dims. (unlike getSlice/get, which return a bare typed array).
    val get_slice_nd(const std::string& key, val windows) const {
        const std::vector<size_t> full_shape = m_ds->shape_of(key);
        // sub-slice calculation (with helper)
        const std::vector<Slice> slices = parse_slices(full_shape, windows);
        return dispatch_numeric(m_ds->dtype_of(key), "getSliceND", [&](auto tag, const char* ctor) {
            NDArray<decltype(tag)> arr = m_ds->get_slice<decltype(tag)>(key, slices);
            // Create and fill `out` object. { data, shape }
            val out = val::object();
            out.set("data", to_typed_array(arr, ctor));
            val out_shape = val::array();
            const std::vector<size_t>& os = arr.shape();
            for (size_t i = 0; i < os.size(); ++i) out_shape.set(i, val(static_cast<double>(os[i])));
            out.set("shape", out_shape);
            return out;
        });
    }

    // Same windowed strided read as getSliceND, but returns NDArray
    // (a shape-carrying handle you must .delete()) instead of a { data, shape }
    // object — the getArray()-style counterpart to getSliceND()'s get()-style result.
    JsNDArray get_slice_ndarray(const std::string& key, val windows) const {
        const std::vector<size_t> full_shape = m_ds->shape_of(key);
        const std::vector<Slice> slices = parse_slices(full_shape, windows);
        return dispatch_numeric(m_ds->dtype_of(key), "getSliceNDArray",
            [&](auto tag, const char*) -> JsNDArray {
                return JsNDArray(ValueVariant(m_ds->get_slice<decltype(tag)>(key, slices)));
            });
    }

    // Read the same window from three 1-D arrays and return it interleaved as one
    // Float32Array [x0,y0,z0, x1,y1,z1, ...].
    //
    // This is the shape GPU vertex buffers want, and doing the interleave here saves
    // the caller three separate heap->JS copies plus a JS-side transpose per batch —
    // which matters when a batch is hundreds of thousands of points. The arrays are
    // typically float64 on disk (full precision positions); float32 is what the
    // renderer uploads anyway.
    val get_slice_xyz_f32(const std::string& kx, const std::string& ky,
                          const std::string& kz, double start, double count) const {
        // Size from the CLAMPED window, so a request that runs past the end yields a
        // short array rather than one padded with zeros the caller would render.
        const size_t n = slice_1d(kx, start, count).length();
        std::vector<float> out(n * 3);
        const std::string* keys[3] = {&kx, &ky, &kz};
        for (int axis = 0; axis < 3; ++axis) {
            const std::string& key = *keys[axis];
            const std::vector<Slice> s = {slice_1d(key, start, count)};
            switch (m_ds->dtype_of(key)) {
                case DataType::FLOAT64: scatter(out, axis, m_ds->get_slice<double>(key, s)); break;
                case DataType::FLOAT32: scatter(out, axis, m_ds->get_slice<float>(key, s));  break;
                case DataType::INT32:   scatter(out, axis, m_ds->get_slice<int32_t>(key, s)); break;
                case DataType::INT16:   scatter(out, axis, m_ds->get_slice<int16_t>(key, s)); break;
                default:
                    throw std::runtime_error("getSliceXYZ(): unsupported dtype for key " + key);
            }
        }
        val view(typed_memory_view(out.size(), out.data()));
        return val::global("Float32Array").new_(view);
    }

    // Requests issued so far (for demos/tests) — proves reads hit the network.
    double network_requests() const {
        return static_cast<double>(star::g_network_request_count.load());
    }

    // --- introspection --------------------------------------------------------

    // True if `key` exists in EITHER namespace — a stored array or a metadata-block
    // value. (For an array-only or metadata-only test, use keys()/metaContains.)
    bool contains(const std::string& key) const { return m_ds->contains(key); }

    // Length of array `key` along its FIRST dimension (like len() of a numpy array —
    // rows, not total elements). Use shape() for the full dims / element count.
    double array_length(const std::string& key) const {
        return static_cast<double>(m_ds->array_length(key));
    }

    // Number of array entries / of metadata-block entries.
    double size() const { return static_cast<double>(m_ds->size()); }
    double meta_count() const { return static_cast<double>(m_ds->get_metadata_count()); }

    bool is_read_only() const { return m_ds->is_read_only(); }
    std::string filename() const { return m_ds->get_filename(); }

    // Parsed file header as a plain JS object.
    val file_header() const {
        const star::FileHeader& h = m_ds->get_file_header();
        val out = val::object();
        out.set("magic", std::string(h.magic, sizeof(h.magic)));
        out.set("formatVersion", static_cast<double>(h.format_version));
        out.set("headerSize", static_cast<double>(h.header_size));
        out.set("entryCount", static_cast<double>(h.entry_count));
        out.set("layerCount", static_cast<double>(h.layer_count));
        out.set("keyRegistryCount", static_cast<double>(h.key_registry_count));
        out.set("versionString", h.getVersionString());
        return out;
    }

    // Warm the cache for `keys` (an array of key names) in one batch. Over a remote
    // source this issues the covering ranged GETs up front (parallel where possible)
    // instead of lazily on first access. Throws if any key is unknown.
    void prefetch(val keys) {
        std::vector<std::string> ks;
        const unsigned n = keys["length"].as<unsigned>();
        ks.reserve(n);
        for (unsigned i = 0; i < n; ++i) ks.push_back(keys[i].as<std::string>());
        m_ds->prefetch(ks);
    }

    // --- layers ---------------------------------------------------------------

    // View an existing layer (throws if absent) / create a new one (throws if it
    // already exists). Both return a Layer handle the caller must .delete().
    JsLayer get_layer(const std::string& name) const { return JsLayer(m_ds->get_layer(name)); }
    JsLayer create_layer(const std::string& name) { return JsLayer(m_ds->create_layer(name)); }

    bool has_layer(const std::string& name) const { return m_ds->has_layer(name); }

    val list_layers() const {
        std::vector<std::string> ls = m_ds->list_layers();
        val out = val::array();
        for (size_t i = 0; i < ls.size(); ++i) out.set(i, val(ls[i]));
        return out;
    }

    // Base-layer inheritance for layer lookups (OpenOptions.layer_inheritance; off by
    // default). When on, a key missing from a layer resolves to the base layer's
    // value in Layer.get()/keys()/has(); when off, a layer miss stays a miss.
    bool layer_inheritance() const { return m_ds->layer_inheritance(); }
    void set_layer_inheritance(bool on) { m_ds->set_layer_inheritance(on); }

private:
    // Clamp a [start, count) request to the array's actual length, so a caller that
    // asks for one batch past the end gets a short (or empty) result rather than an
    // out-of-range throw.
    Slice slice_1d(const std::string& key, double start, double count) const {
        const std::vector<size_t> shape = m_ds->shape_of(key);
        if (shape.size() != 1) {
            throw std::runtime_error("slice: key '" + key + "' is not 1-D");
        }
        const size_t n = shape[0];
        const size_t begin = start <= 0 ? 0 : std::min(static_cast<size_t>(start), n);
        const size_t want = count <= 0 ? 0 : static_cast<size_t>(count);
        return Slice{begin, std::min(begin + want, n), 1};
    }

    // Write arr[i] into out[i * 3 + axis], converting to float.
    template <typename T>
    static void scatter(std::vector<float>& out, int axis, const NDArray<T>& arr) {
        const std::vector<T>& d = arr.data();
        const size_t n = std::min(d.size(), out.size() / 3);
        for (size_t i = 0; i < n; ++i) out[i * 3 + axis] = static_cast<float>(d[i]);
    }

    std::shared_ptr<StarDataset> m_ds;
};

// Create a NEW writable dataset at `path` with an explicit StarConfig (compression,
// block size, metadata-block options), returning the JS handle. Unlike the
// Dataset(path,"w") constructor — which creates on flush with DEFAULT config —
// create() lets the caller pick the write-time codec. An existing file at `path` is
// overwritten. Note: on WASM only NONE and the GZIP* codecs are runnable, no ZLIB.
JsDataset create_dataset(const std::string& path, const StarConfig& config) {
    return JsDataset(StarDataset::create(path, config));
}

// Open a READ-ONLY dataset from a complete .stards image already in memory —
// `bytes` is a JS Uint8Array (or any numeric TypedArray/Array of byte values). No
// filesystem or network is touched: the bytes are copied into the Wasm heap and
// parsed. Round-trips with writeBytes() — openBytes(ds.writeBytes()) reconstructs
// the dataset. Throws if the bytes are not a valid STAR image. (An ArrayBuffer has
// no length/indexing, so wrap it first: new Uint8Array(buf).)
JsDataset open_bytes_dataset(val bytes, val opts) {
    std::vector<uint8_t> buf = convertJSArrayToNumberVector<uint8_t>(bytes);
    return JsDataset(StarDataset::open_bytes(buf.data(), buf.size(), parse_open_options(opts)));
}

// --- module-level (non-Dataset) helpers ---------------------------------------

std::string library_version() { return star::getLibraryVersion(); }

// Process-wide network-request counter (the same one Dataset.networkRequests()
// reads). Exposed at module scope so callers can reset it between operations.
double network_request_count() { return static_cast<double>(star::getNetworkRequestCount()); }
void reset_network_request_count() { star::resetNetworkRequestCount(); }

// Size in bytes of one element of the named dtype ("int32", "float64", ...).
double dtype_size(const std::string& name) {
    return static_cast<double>(star::datatype_size(datatype_from_string(name)));
}

}  // namespace

EMSCRIPTEN_BINDINGS(stards) {
    class_<JsDataset>("Dataset")
        .constructor<std::string>()
        .constructor<std::string, std::string>()
        .constructor<std::string, std::string, val>()
        .function("keys", &JsDataset::keys)
        .function("dtype", &JsDataset::dtype)
        .function("shape", &JsDataset::shape)
        .function("get", &JsDataset::get)
        .function("put", &JsDataset::put)
        .function("getArray", &JsDataset::get_array)
        .function("putArray", &JsDataset::put_array)
        .function("flush", &JsDataset::flush)
        .function("writeBytes", &JsDataset::write_bytes)
        .function("saveTo", &JsDataset::save_to)
        .function("close", &JsDataset::close)
        .function("metaString", &JsDataset::meta_string)
        .function("metaGet", &JsDataset::meta_get)
        .function("metaKeys", &JsDataset::meta_keys)
        .function("metaContains", &JsDataset::meta_contains)
        .function("metaDtype", &JsDataset::meta_dtype)
        .function("metaShape", &JsDataset::meta_shape)
        .function("metaGetAll", &JsDataset::meta_get_all)
        .function("metaPut", &JsDataset::meta_put)
        .function("metaPutArray", &JsDataset::meta_put_array)
        .function("metaRemove", &JsDataset::meta_remove)
        .function("metaClear", &JsDataset::meta_clear)
        .function("isSliceable", &JsDataset::is_sliceable)
        .function("getSlice", &JsDataset::get_slice)
        .function("getSliceND", &JsDataset::get_slice_nd)
        .function("getSliceNDArray", &JsDataset::get_slice_ndarray)
        .function("getSliceXYZ", &JsDataset::get_slice_xyz_f32)
        .function("contains", &JsDataset::contains)
        .function("arrayLength", &JsDataset::array_length)
        .function("size", &JsDataset::size)
        .function("metaCount", &JsDataset::meta_count)
        .function("isReadOnly", &JsDataset::is_read_only)
        .function("filename", &JsDataset::filename)
        .function("fileHeader", &JsDataset::file_header)
        .function("prefetch", &JsDataset::prefetch)
        .function("getLayer", &JsDataset::get_layer)
        .function("createLayer", &JsDataset::create_layer)
        .function("hasLayer", &JsDataset::has_layer)
        .function("listLayers", &JsDataset::list_layers)
        .function("layerInheritance", &JsDataset::layer_inheritance)
        .function("setLayerInheritance", &JsDataset::set_layer_inheritance)
        .function("networkRequests", &JsDataset::network_requests);

    class_<JsLayer>("Layer")
        .function("name", &JsLayer::name)
        .function("keys", &JsLayer::keys)
        .function("contains", &JsLayer::contains)
        .function("get", &JsLayer::get)
        .function("put", &JsLayer::put)
        .function("getArray", &JsLayer::get_array)
        .function("putArray", &JsLayer::put_array)
        .function("metaKeys", &JsLayer::meta_keys)
        .function("metaContains", &JsLayer::meta_contains)
        .function("metaGet", &JsLayer::meta_get)
        .function("metaPut", &JsLayer::meta_put)
        .function("metaPutArray", &JsLayer::meta_put_array)
        .function("metaRemove", &JsLayer::meta_remove);

    // First-class, dtype-erased N-D array. Construct from JS data
    // (new Module.NDArray(value, shape, dtype)) or a factory; read it out with
    // data()/at(); write it into a dataset with Dataset/Layer.putArray().
    class_<JsNDArray>("NDArray")
        .constructor<val, val, std::string>()
        .function("dtype", &JsNDArray::dtype)
        .function("shape", &JsNDArray::shape)
        .function("size", &JsNDArray::size)
        .function("ndim", &JsNDArray::ndim)
        .function("data", &JsNDArray::data)
        .function("at", &JsNDArray::at)
        .function("reshape", &JsNDArray::reshape)
        .class_function("zeros", &JsNDArray::zeros)
        .class_function("ones", &JsNDArray::ones)
        .class_function("full", &JsNDArray::full);

    // Compression codecs for StarConfig.compression / .metadataCompression. Only
    // codecs this build can actually run are exposed.  On WASM that's NONE + the 
    // GZIP* variants (zlib). The _BLOCK shuffle variants stay sliceable;
    // the plain _SHUFFLE ones are legacy whole-array (not sliceable).
    enum_<CompressionAlgorithm>("Compression")
        .value("NONE", CompressionAlgorithm::NONE)
#ifdef ENABLE_ZLIB
        .value("GZIP", CompressionAlgorithm::GZIP)
        .value("GZIP_SHUFFLE", CompressionAlgorithm::GZIP_SHUFFLE)
        .value("GZIP_SHUFFLE_BLOCK", CompressionAlgorithm::GZIP_SHUFFLE_BLOCK)
#endif
#ifdef ENABLE_ZSTD
        .value("ZSTD", CompressionAlgorithm::ZSTD)
#endif
#ifdef ENABLE_LZ4
        .value("LZ4", CompressionAlgorithm::LZ4)
        .value("LZ4_SHUFFLE", CompressionAlgorithm::LZ4_SHUFFLE)
        .value("LZ4_SHUFFLE_BLOCK", CompressionAlgorithm::LZ4_SHUFFLE_BLOCK)
#endif
        ;

    // Write-time configuration for create(). Constructed with defaults; set only
    // the fields you want to change. (metadata_force_separate_keys and the buffer/
    // arena tuning knobs are intentionally not exposed yet.)
    class_<StarConfig>("StarConfig")
        .constructor<>()
        .property("compression", &StarConfig::compression)
        .property("blockSize", &StarConfig::block_size)
        .property("metadataBlockEnabled", &StarConfig::metadata_block_enabled)
        .property("metadataMaxBlockSize", &StarConfig::metadata_max_block_size)
        .property("metadataCompression", &StarConfig::metadata_compression);

    // Module.create(path, config) -> Dataset. See create_dataset() above.
    function("create", &create_dataset);
    // Module.openBytes(uint8Array) -> read-only Dataset. See open_bytes_dataset().
    function("openBytes", &open_bytes_dataset);

    // Module-level helpers.
    function("libraryVersion", &library_version);
    function("networkRequestCount", &network_request_count);
    function("resetNetworkRequestCount", &reset_network_request_count);
    function("dtypeSize", &dtype_size);
}
