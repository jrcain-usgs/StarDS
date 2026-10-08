// getSliceND test: N-dimensional strided windows over 1-D/2-D/3-D arrays, checked
// against a JS reference implementation of row-major strided slicing.
//
// Build first, then run from the repo root:
//   cmake --build build --target wasm_tests
//   node build/tests/slicing.mjs
import { loadStarDS } from '../stards.mjs';

const { Dataset } = await loadStarDS();
let fails = 0;
const ok = (label, cond, extra = '') => {
  if (!cond) fails++;
  console.log(`${cond ? 'OK ' : 'XX '} ${label}${extra ? '  ' + extra : ''}`);
};
const j = (x) => JSON.stringify(x);

// Reference row-major strided slice, mirroring StarDataset::get_slice semantics:
// per-dim [start,stop,step) with clamping, output dims = ceil((stop-start)/step).
function refSlice(flat, shape, windows) {
  const W = shape.map((d, i) => {
    const w = windows[i];
    let s = 0, e = d, st = 1;
    if (Array.isArray(w)) { s = w[0] ?? 0; e = w[1] ?? d; st = w[2] ?? 1; }
    else if (w && typeof w === 'object') { s = w.start ?? 0; e = w.stop ?? d; st = w.step ?? 1; }
    s = Math.max(0, Math.min(s, d));
    e = Math.max(s, Math.min(e, d));
    st = Math.max(1, st);
    return [s, e, st];
  });
  const outShape = W.map(([s, e, st]) => Math.max(0, Math.ceil((e - s) / st)));
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  const out = [];
  const rec = (dim, offset) => {
    if (dim === shape.length) { out.push(flat[offset]); return; }
    const [s, e, st] = W[dim];
    for (let idx = s; idx < e; idx += st) rec(dim + 1, offset + idx * strides[dim]);
  };
  rec(0, 0);
  return { data: out, shape: outShape };
}

function check(label, r, key, flat, shape, windows) {
  const got = r.getSliceND(key, windows);
  const want = refSlice(flat, shape, windows);
  const okData = j([...got.data]) === j(want.data);
  const okShape = j([...got.shape]) === j(want.shape);
  ok(`${label}: shape ${j([...got.shape])}`, okShape, okShape ? '' : `want ${j(want.shape)}`);
  ok(`${label}: data`, okData, okData ? '' : `got ${j([...got.data])} want ${j(want.data)}`);
}

// --- author 1-D / 2-D / 3-D arrays --------------------------------------------
const vec = Float64Array.from({ length: 10 }, (_, i) => i); // 0..9
const mat = Float64Array.from({ length: 12 }, (_, i) => i); // 3x4
const vol = Int32Array.from({ length: 24 }, (_, i) => i);   // 2x3x4

const w = await new Dataset('nd.stards', 'w');
w.put('vec', vec, [10], 'float64');
w.put('mat', mat, [3, 4], 'float64');
w.put('vol', vol, [2, 3, 4], 'int32');
w.flush();
w.close();
w.delete();

const r = await new Dataset('nd.stards', 'r');

// 1-D: strided, and equivalence with the 1-D getSlice(start,count).
check('vec strided [2,9,2]', r, 'vec', vec, [10], [[2, 9, 2]]);
ok('vec getSliceND == getSlice', j([...r.getSliceND('vec', [[3, 7]]).data]) === j([...r.getSlice('vec', 3, 4)]));

// 2-D: sub-block, column step, and "fewer windows -> trailing dim full".
check('mat rows{0,3,2} cols{1,4,1}', r, 'mat', mat, [3, 4], [[0, 3, 2], [1, 4, 1]]);
check('mat cols step 2', r, 'mat', mat, [3, 4], [[0, 3, 1], [0, 4, 2]]);
check('mat one window (rows 1..3, all cols)', r, 'mat', mat, [3, 4], [[1, 3]]);
check('mat full (no windows)', r, 'mat', mat, [3, 4], []);
check('mat clamped past end', r, 'mat', mat, [3, 4], [[0, 99, 1], [2, 99, 1]]);

// 3-D.
check('vol block', r, 'vol', vol, [2, 3, 4], [[0, 2, 1], [1, 3, 1], [0, 4, 2]]);

// getSliceNDArray: same windowed read, but returns an NDArray handle.
{
  const win = [[0, 3, 2], [1, 4, 1]]; // matches 'mat rows{0,3,2} cols{1,4,1}'
  const flat = r.getSliceND('mat', win);        // { data, shape }
  const nd = r.getSliceNDArray('mat', win);     // NDArray
  ok('getSliceNDArray shape matches getSliceND', j([...nd.shape()]) === j([...flat.shape]));
  ok('getSliceNDArray data matches getSliceND', j([...nd.data()]) === j([...flat.data]));
  ok('getSliceNDArray dtype', nd.dtype() === 'float64');
  ok('getSliceNDArray at([0,2])', nd.at([0, 2]) === flat.data[2]); // first row, third selected col
  nd.delete();
}

// error: more windows than dims (both variants).
try {
  r.getSliceND('vec', [[0, 5], [0, 5]]);
  ok('too many windows throws', false);
} catch (e) {
  ok('too many windows throws', true, '-> ' + e.message);
}
try {
  r.getSliceNDArray('vec', [[0, 5], [0, 5]]);
  ok('getSliceNDArray too many windows throws', false);
} catch (e) {
  ok('getSliceNDArray too many windows throws', true, '-> ' + e.message);
}

r.delete();
console.log(fails ? `\n${fails} FAILURE(S)` : '\nALL PASS');
process.exit(fails ? 1 : 0);
