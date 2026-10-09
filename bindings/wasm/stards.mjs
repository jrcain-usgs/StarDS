// Loader for StarDS Wasm.
// Makes C++ exceptions legible from JavaScript.
//
// Usage:
//   import { loadStarDS } from '../stards.mjs';
//   const { Dataset, Module, decodeException } = await loadStarDS();
//   const ds = await Dataset('https://.../foo.stards');   // or new Dataset(...)
//   try { ds.get('missing'); } catch (e) { console.error(e.message); }
//
// Writing to S3 needs AWS credentials. Pass via the `env` option 
// (installed in preRun, before the runtime reads them):
//   const { Dataset } = await loadStarDS({
//     env: { AWS_ACCESS_KEY_ID: '…', AWS_SECRET_ACCESS_KEY: '…',
//            AWS_DEFAULT_REGION: 'us-east-1' },   // AWS_SESSION_TOKEN for STS
//   });
//   const ds = await new Dataset('s3://bucket/key.stards', 'w');
//   ds.put('a', new Float32Array([1,2,3])); await ds.flush();
// Any other Module init options (preRun, print, locateFile, …) pass through too.

import initStarDS from './stards_wasm.mjs';


// Decodes C++ errors into JS ones.
export function decodeException(Module, e) {
  if (typeof e === 'number' && Module.getExceptionMessage) {
    try { return new Error(Module.getExceptionMessage(e).join(': ')); }
    catch { return new Error('wasm exception #' + e); }
  }
  if (Array.isArray(e)) return new Error(e.join(': '));
  return e;
}

// TypedArray constructor name -> StarDS dtype
const TYPED_ARRAY_DTYPE = {
  Int8Array: 'int8', Int16Array: 'int16', Int32Array: 'int32',
  Uint8Array: 'uint8', Uint8ClampedArray: 'uint8', Uint16Array: 'uint16', Uint32Array: 'uint32',
  Float32Array: 'float32', Float64Array: 'float64',
  BigInt64Array: 'int64', BigUint64Array: 'uint64',
};

// Infer the on-disk dtype from a JS value when the caller didn't pass one.
function inferDtype(value) {
  const ctorName = value?.constructor?.name;
  if (ctorName && TYPED_ARRAY_DTYPE[ctorName]) return TYPED_ARRAY_DTYPE[ctorName];
  if (typeof value === 'string') return 'string';
  if (typeof value === 'bigint') return 'int64';
  if (typeof value === 'number') return 'float64';
  if (Array.isArray(value)) {
    if (value.length === 0) return 'float64';
    const first = value[0];
    if (typeof first === 'string') return 'string';
    if (typeof first === 'bigint') return 'int64';
    return 'float64'; // numbers
  }
  throw new Error(`cannot infer dtype from value of type ${ctorName ?? typeof value}; pass an explicit dtype`);
}

// If an array's first element is array-like, e.g. [[1,2,3],[4,5,6]],
// shape/flat data can be derived from the nesting; passed shape is ignored.
function isNested(v) {
  return Array.isArray(v) && v.length > 0 && (Array.isArray(v[0]) || ArrayBuffer.isView(v[0]));
}

// Descend the first element per axis to get
// {shape, container (used to infer dtype), scalar (1st leaf value)}:
function nestedInfo(v) {
  const shape = [];
  let cur = v;
  let container = v;
  while (Array.isArray(cur) || ArrayBuffer.isView(cur)) {
    shape.push(cur.length);
    container = cur;
    cur = cur[0];
  }
  return { shape, container, scalar: cur };
}

// Flatten a nested array depth-first into `out` (row-major). O(n).
function flattenInto(v, out) {
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    for (let i = 0; i < v.length; i++) flattenInto(v[i], out);
  } else {
    out.push(v);
  }
  return out;
}

// Resolve (value, shape, dtype) for put()/NDArray(): flatten a nested array (shape
// derived from nesting, dtype from its innermost container/leaf); otherwise pass the
// value through with an omitted shape -> null (1-D) and an omitted dtype -> inferred.
function normalizeData(value, shape, dtype) {
  if (isNested(value)) {
    const { shape: derived, container, scalar } = nestedInfo(value);
    const dt = dtype
      ?? (ArrayBuffer.isView(container)
            ? (TYPED_ARRAY_DTYPE[container.constructor.name] ?? 'float64')
            : inferDtype(scalar));
    return [flattenInto(value, []), derived, dt];
  }
  return [value, shape ?? null, dtype ?? inferDtype(value)];
}


function isNDArray(Module, x) {
  return x instanceof Module.NDArray;
}

// Polymorphic put(key, value, shape?, dtype?): an NDArray goes straight to the C++
// putArray; anything else is normalized (dtype inferred, shape defaulted, nested
// flattened) and sent to the flat put. `target` is the raw (un-proxied) instance.
function invokePut(Module, target, key, value, shape, dtype) {
  if (isNDArray(Module, value)) return target.putArray(key, value);
  const [v, s, dt] = normalizeData(value, shape, dtype);
  return target.put(key, v, s, dt);
}

// Polymorphic metaPut(key, value, dtype?). The metadata put path has no shape arg,
// so an NDArray (or a nested array, via a temporary NDArray) routes to metaPutArray
// for N-D; a scalar/1-D value uses the flat metaPut.
function invokeMetaPut(Module, target, key, value, dtype) {
  if (isNDArray(Module, value)) return target.metaPutArray(key, value);
  if (isNested(value)) {
    const [v, s, dt] = normalizeData(value, null, dtype);
    const nd = new Module.NDArray(v, s, dt);
    try { return target.metaPutArray(key, nd); } finally { nd.delete(); }
  }
  return target.metaPut(key, value, dtype ?? inferDtype(value));
}

// An embind handle is a JS object exposing a .delete() (e.g. a returned Layer).
// Wrap those too so their methods get the same error decoding; leave plain values
// (typed arrays, {data,shape}, fileHeader objects, primitives) untouched.
function maybeWrap(Module, x) {
  if (x && typeof x === 'object' && typeof x.delete === 'function') return wrapInstance(Module, x);
  return x;
}

// Proxy an embind instance so each method call decodes a sync throw or an async
// rejection, and any returned embind handle (e.g. from getLayer) is itself wrapped.
// `this` stays bound to the real object (methods applied on target), so .delete()
// and the rest behave normally.
function wrapInstance(Module, obj) {
  return new Proxy(obj, {
    get(target, prop, receiver) {
      // The NDArray-only entry points are folded into put()/metaPut(); hide them.
      if (prop === 'putArray' || prop === 'metaPutArray') return undefined;
      const v = Reflect.get(target, prop, receiver);
      if (typeof v !== 'function') return v;
      return (...args) => {
        try {
          let r;
          // put()/metaPut() are polymorphic: they accept raw values or an NDArray.
          if (prop === 'put') r = invokePut(Module, target, ...args);
          else if (prop === 'metaPut') r = invokeMetaPut(Module, target, ...args);
          else r = v.apply(target, args);
          if (r && typeof r.then === 'function') {
            return r.then((x) => maybeWrap(Module, x), (e) => { throw decodeException(Module, e); });
          }
          return maybeWrap(Module, r);
        } catch (e) {
          throw decodeException(Module, e);
        }
      };
    },
  });
}

// options: an Emscripten Module config object forwarded to the WASM factory
// (preRun, print, locateFile, wasmBinary, …), plus one extra key:
//   env — { NAME: value, … } environment variables to set on Module.ENV before
//         the runtime runs, e.g. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
//         AWS_SESSION_TOKEN / AWS_DEFAULT_REGION for S3 writes, or
//         AWS_S3_ENDPOINT / AWS_VIRTUAL_HOSTING / AWS_HTTPS for S3-compatible
//         stores.
//         NOTE: Don't pass empty strings for credential or region vars.
export async function loadStarDS(options = {}) {
  const { env, preRun, ...moduleConfig } = options;

  const userPreRun = preRun == null ? [] : (Array.isArray(preRun) ? preRun : [preRun]);
  const envPreRun = (mod) => {
    if (!env) return;
    for (const [k, v] of Object.entries(env)) {
      if (v != null) mod.ENV[k] = String(v);
    }
  };

  const Module = await initStarDS({ ...moduleConfig, preRun: [envPreRun, ...userPreRun] });
  const RawDataset = Module.Dataset;

  // A Dataset factory whose instances rethrow decoded Errors. The embind
  // constructor may itself return a Promise (ASYNCIFY open), so handle both a
  // resolved instance and a synchronous one. Callable with or without `new`.
  function Dataset(...args) {
    try {
      const inst = new RawDataset(...args);
      if (inst && typeof inst.then === 'function') {
        return inst.then(
          (i) => wrapInstance(Module, i),
          (e) => { throw decodeException(Module, e); },
        );
      }
      return wrapInstance(Module, inst);
    } catch (e) {
      throw decodeException(Module, e);
    }
  }

  // create(path, config) -> a wrapped Dataset (same error-decoding as above).
  // config is a Module.StarConfig instance; compression uses Module.Compression.
  function create(path, config) {
    try {
      const inst = Module.create(path, config);
      if (inst && typeof inst.then === 'function') {
        return inst.then(
          (i) => wrapInstance(Module, i),
          (e) => { throw decodeException(Module, e); },
        );
      }
      return wrapInstance(Module, inst);
    } catch (e) {
      throw decodeException(Module, e);
    }
  }

  // openBytes(uint8Array, opts?) -> a wrapped read-only Dataset (same error
  // decoding). opts is an optional OpenOptions object, e.g. { layerInheritance: true }.
  function openBytes(bytes, opts = {}) {
    try {
      const inst = Module.openBytes(bytes, opts);
      if (inst && typeof inst.then === 'function') {
        return inst.then(
          (i) => wrapInstance(Module, i),
          (e) => { throw decodeException(Module, e); },
        );
      }
      return wrapInstance(Module, inst);
    } catch (e) {
      throw decodeException(Module, e);
    }
  }

  // Wrap a module-level free function so its throws decode too (only dtypeSize can
  // throw, but wrap all for uniformity).
  const wrapFn = (fn) => (...args) => {
    try {
      const r = fn(...args);
      if (r && typeof r.then === 'function') return r.then((x) => x, (e) => { throw decodeException(Module, e); });
      return r;
    } catch (e) {
      throw decodeException(Module, e);
    }
  };

  // NDArray(value, shape?, dtype?) -> a wrapped NDArray handle (decoded errors), with
  // wrapped .zeros/.ones/.full factories attached. `shape` defaults to 1-D of the
  // data length and `dtype` is inferred from `value` when omitted (see inferDtype).
  // A nested array (e.g. [[1,2,3],[4,5,6]]) is flattened with its shape derived from
  // the nesting (any passed shape is ignored).
  const NDArray = (value, shape, dtype) =>
    wrapInstance(Module, (() => {
      const [v, s, dt] = normalizeData(value, shape, dtype);
      try { return new Module.NDArray(v, s, dt); }
      catch (e) { throw decodeException(Module, e); }
    })());
  // Factories have no data to infer from, so dtype defaults to 'float64'.
  NDArray.zeros = (shape, dtype = 'float64') => wrapInstance(Module, wrapFn(() => Module.NDArray.zeros(shape, dtype))());
  NDArray.ones = (shape, dtype = 'float64') => wrapInstance(Module, wrapFn(() => Module.NDArray.ones(shape, dtype))());
  NDArray.full = (shape, value, dtype = 'float64') => wrapInstance(Module, wrapFn(() => Module.NDArray.full(shape, value, dtype))());

  return {
    Module,
    Dataset,
    create,
    openBytes,
    NDArray,
    libraryVersion: wrapFn(Module.libraryVersion),
    networkRequestCount: wrapFn(Module.networkRequestCount),
    resetNetworkRequestCount: wrapFn(Module.resetNetworkRequestCount),
    dtypeSize: wrapFn(Module.dtypeSize),
    decodeException: (e) => decodeException(Module, e),
  };
}
