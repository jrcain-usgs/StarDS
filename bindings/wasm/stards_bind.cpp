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

// ---- NDArray ----

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
    // value: TypedArray/Array/string;
    // shape: array-like, [] = scalar, or null = 1-D of the data length.
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


    // Read a single element at the given indices
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

    // Reshape in place (total element count must match).
    void reshape(val new_shape) {
        std::vector<size_t> s = to_size_vector(new_shape);
        std::visit([&](auto& arr) { arr.reshape(s); }, m_var);
    }

    // NDArray Factories
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


// JS handle for LayerView. Holds a shared_ptr so it keeps the base
// dataset alive independently of the Dataset handle it came from.
class JsLayer {
public:
    explicit JsLayer(std::shared_ptr<LayerView> layer) : m_layer(std::move(layer)) {}

    std::string name() const { return m_layer->name(); }

    // Keys in this layer, as string array
    val keys() const {
        std::vector<std::string> k = m_layer->keys();
        val out = val::array();
        for (size_t i = 0; i < k.size(); ++i) out.set(i, val(k[i]));
        return out;
    }

    // Check if layer contains this key
    bool contains(const std::string& key) const { return m_layer->contains(key); }

    // Read array `key` from this layer.
    // Numeric -> typed array; STRING -> Array<string>.
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

    // Write array `key` into this layer. Requires dataset in write mode.
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

    // Read array `key` as NDArray (shape-aware handle, must .delete()).
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
    // Store an NDArray as layer metadata.
    void meta_put_array(const std::string& key, const JsNDArray& arr) {
        std::visit([&](const auto& a) { m_layer->meta.put(key, a); }, arr.variant());
    }
    void meta_remove(const std::string& key) { m_layer->meta.remove(key); }

private:
    // Resolve the dtype of a layer array key.
    // At time of writing, no dtype implemented for C++ LayerView.
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

class JsDataset {
public:
    // Opens read-only.
    explicit JsDataset(const std::string& path)
        : m_ds(StarDataset::open(path, "r")) {}

    // Opens with file mode ("r", "w"/"rw"/"a").
    JsDataset(const std::string& path, const std::string& mode)
        : m_ds(StarDataset::open(path, mode)) {}

    // Opens with file mode and OpenOptions. Example:
    // new Dataset(url, "r", { layerInheritance: true, prefetchWholeBelowBytes: 0 }).
    JsDataset(const std::string& path, const std::string& mode, val opts)
        : m_ds(StarDataset::open(path, mode, parse_open_options(opts))) {}

    // Wrap an already-constructed dataset (used by the create() factory below).
    // Not registered as a JS constructor.
    explicit JsDataset(std::shared_ptr<StarDataset> ds) : m_ds(std::move(ds)) {}

    // Get an array of keys present in the dataset
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

    // Writes an array to `key` from JS types (value (flat JS Array), shape, dtype)
    // Use `put_array() for NDArrays.
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

    // Read array `key` as a NDArray (NDArrays require .delete()).
    // Use `get()` for a flat JS array.
    JsNDArray get_array(const std::string& key) const { return JsNDArray(read_variant(*m_ds, key)); }


    // Writes NDArray to `key`.  Use `put` to work in plain JS types.
    void put_array(const std::string& key, JsNDArray arr) {
        std::visit([&](auto&& a) { m_ds->put(key, std::move(a)); }, arr.take());
    }

    // Persist pending writes to the WASM virtual filesystem. 
    // Use `writeBytes()` to instead get the image back as bytes.
    void flush() { m_ds->flush(); }

    // Serialize dataset to .stards image and return it as a Uint8Array. 
    // Just returns bytes, doesn't touch source file.
    val write_bytes() {
        std::vector<char> bytes = m_ds->write_bytes();
        val view(typed_memory_view(bytes.size(),
                                   reinterpret_cast<const uint8_t*>(bytes.data())));
        return val::global("Uint8Array").new_(view);
    }

    // Persist the dataset to `path` in WASM virtual filesystem.
    // Makes a second copy, like "Save As..." in many apps.
    void save_to(const std::string& path) { m_ds->save_to(path); }

    // Flush + release. The JS handle itself must still be .delete()'d.
    void close() { m_ds->close(); }



    // Reads string from 1st element of metadata array.
    // Returns "" if no string present.
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

    // Returns the value for a metadata `key`
    val meta_get(const std::string& key) const {
        std::shared_ptr<MetadataValue> mv = m_ds->meta.get(key);
        if (!mv) return val::null();
        return meta_value_to_js(*mv);
    }

    // Returns array of metadata keys
    val meta_keys() const {
        std::vector<std::string> k = m_ds->get_metadata_keys();
        val out = val::array();
        for (size_t i = 0; i < k.size(); ++i) out.set(i, val(k[i]));
        return out;
    }

    bool meta_contains(const std::string& key) const { return m_ds->meta.contains(key); }

    // get dtype for key ("int32", "float64", "string", ...)
    // or "" if no key found.
    std::string meta_dtype(const std::string& key) const {
        std::shared_ptr<MetadataValue> mv = m_ds->meta.get(key);
        return mv ? mv->type_name() : std::string();
    }

    // Shape [] for metadata entry (empty for a scalar, [] if absent).
    val meta_shape(const std::string& key) const {
        val out = val::array();
        std::shared_ptr<MetadataValue> mv = m_ds->meta.get(key);
        if (mv) {
            for (size_t i = 0; i < mv->shape.size(); ++i)
                out.set(i, val(static_cast<double>(mv->shape[i])));
        }
        return out;
    }

    // All metadata as a plain JS object { key: value }.
    val meta_get_all() const {
        std::map<std::string, MetadataValue> all = m_ds->meta.get_all();
        val obj = val::object();
        for (const auto& [k, mv] : all) obj.set(k, meta_value_to_js(mv));
        return obj;
    }

    // Write a metadata entry.  Requires writable dataset.
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

    // Store an NDArray as metadata (any shape).
    void meta_put_array(const std::string& key, const JsNDArray& arr) {
        std::visit([&](const auto& a) { m_ds->meta.put(key, a); }, arr.variant());
    }

    void meta_remove(const std::string& key) { m_ds->meta.remove(key); }
    void meta_clear() { m_ds->meta.clear(); }

    // True if `key` is stored as blocks and can be windowed with getSlice().
    bool is_sliceable(const std::string& key) const { return m_ds->is_sliceable(key); }

    // Returns subset of a 1-D array.
    val get_slice(const std::string& key, double start, double count) const {
        const std::vector<Slice> s = {slice_1d(key, start, count)};
        return dispatch_numeric(m_ds->dtype_of(key), "getSlice", [&](auto tag, const char* ctor) {
            return to_typed_array(m_ds->get_slice<decltype(tag)>(key, s), ctor);
        });
    }

    /**
     * @brief N-dimensional strided slice
     * 
     * @param key The array to slice
     * @param windows an array of per-dimension/[start,stop,step?] specs.
     *                ie [[5, 15], [5, 15]] for a 2D 10x10 square starting at 5x5.
     * 
     * @return { data, shape } obj
     */
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

    // Same as getSliceND, but returns NDArray (must be .delete()-d).
    JsNDArray get_slice_ndarray(const std::string& key, val windows) const {
        const std::vector<size_t> full_shape = m_ds->shape_of(key);
        const std::vector<Slice> slices = parse_slices(full_shape, windows);
        return dispatch_numeric(m_ds->dtype_of(key), "getSliceNDArray",
            [&](auto tag, const char*) -> JsNDArray {
                return JsNDArray(ValueVariant(m_ds->get_slice<decltype(tag)>(key, slices)));
            });
    }

    // Read the same window from three 1-D arrays and return it interleaved as one
    // Float32Array [x0,y0,z0, x1,y1,z1, ...], for GPU vertex buffers
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

    // Requests issued so far (for demos/tests).
    double network_requests() const {
        return static_cast<double>(star::g_network_request_count.load());
    }

    // True if `key` exists in EITHER namespace — a stored array or a metadata key.
    bool contains(const std::string& key) const { return m_ds->contains(key); }

    // Length of array `key` along its FIRST dimension.
    // Use shape() for the full dims / element count.
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

    // Prefetch a batch of keys.
    void prefetch(val keys) {
        std::vector<std::string> ks;
        const unsigned n = keys["length"].as<unsigned>();
        ks.reserve(n);
        for (unsigned i = 0; i < n; ++i) ks.push_back(keys[i].as<std::string>());
        m_ds->prefetch(ks);
    }

    // --- layers ---

    // View an existing layer.  Caller must .delete();
    JsLayer get_layer(const std::string& name) const { return JsLayer(m_ds->get_layer(name)); }

    // Create a new layer. Caller must .delete();
    JsLayer create_layer(const std::string& name) { return JsLayer(m_ds->create_layer(name)); }

    bool has_layer(const std::string& name) const { return m_ds->has_layer(name); }

    val list_layers() const {
        std::vector<std::string> ls = m_ds->list_layers();
        val out = val::array();
        for (size_t i = 0; i < ls.size(); ++i) out.set(i, val(ls[i]));
        return out;
    }

    // Check layer-inheritance setting
    bool layer_inheritance() const { return m_ds->layer_inheritance(); }

    void set_layer_inheritance(bool on) { m_ds->set_layer_inheritance(on); }

private:
    // Clamp a [start, count) request to array length. (avoids out-of-range throws)
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

// Create a writable dataset at virtual `path` with StarConfig (compression and block options).
// Use instead of Dataset(path,"w") to specify write-time codec (NONE or GZIP*)
JsDataset create_dataset(const std::string& path, const StarConfig& config) {
    return JsDataset(StarDataset::create(path, config));
}

// Open a read-only .stards image from a JS Uint8Array of bytes.
JsDataset open_bytes_dataset(val bytes, val opts) {
    std::vector<uint8_t> buf = convertJSArrayToNumberVector<uint8_t>(bytes);
    return JsDataset(StarDataset::open_bytes(buf.data(), buf.size(), parse_open_options(opts)));
}

// --- module-level (non-Dataset) helpers ---

std::string library_version() { return star::getLibraryVersion(); }

// Process-wide network-request counter
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

    class_<StarConfig>("StarConfig")
        .constructor<>()
        .property("compression", &StarConfig::compression)
        .property("blockSize", &StarConfig::block_size)
        .property("metadataBlockEnabled", &StarConfig::metadata_block_enabled)
        .property("metadataMaxBlockSize", &StarConfig::metadata_max_block_size)
        .property("metadataCompression", &StarConfig::metadata_compression);

    // Module.create(path, config) -> Dataset.
    function("create", &create_dataset);
    // Module.openBytes(uint8Array) -> read-only Dataset.
    function("openBytes", &open_bytes_dataset);

    // Module-level helpers.
    function("libraryVersion", &library_version);
    function("networkRequestCount", &network_request_count);
    function("resetNetworkRequestCount", &reset_network_request_count);
    function("dtypeSize", &dtype_size);
}
