# Slicing

Slicing loads only the portion of an array you ask for, so you can work with
arrays far larger than memory without reading the whole file. Because
compression is [block-based](compression.md), StarDS decompresses only the
blocks a slice touches.

## Reading a slice

Pass a list of `(start, stop)` or `(start, stop, step)` tuples — one per
dimension:

=== "Python"

    ```python
    import numpy as np
    from pystards import StarDataset

    # Store a large array
    with StarDataset.create("large.stards") as ds:
        ds["big_matrix"] = np.random.rand(10000, 10000)

    # Read only a 100×100 corner (efficient!)
    with StarDataset.open("large.stards", mode="r") as ds:
        subset = ds.get_slice("big_matrix", [(0, 100), (0, 100)])
        print(subset.shape)  # (100, 100)
    ```

=== "JS"

    ```js
    import { loadStarDS } from './stards.mjs';
    const { Dataset, NDArray } = await loadStarDS();

    // Open a large Array
    const URL = 'https://asc-isisdata.s3.us-west-2.amazonaws.com/cnf_test_data/largenet.stards';
    const linearReader = await new Dataset(URL);

    // Read a 1-D Slice (use getSlice for convenience)
    console.info('Slice', await linearReader.getSlice('m.sample', 0, 6));

    // Write a 2-D Array
    const rectWriter = await new Dataset('data.stards', 'w');
    rectWriter.put('big-matrix', NDArray.full([512, 512], 3.14, 'float64'));
    rectWriter.flush();
    
    // Read a 2-D Slice
    const rectReader = await new Dataset('data.stards', 'r');
    // 2D Slice from [32, 64] to [48, 96].
    // at 1/4 resolution (step 4 gets every 4th value, step 1 gets full resolution).
    console.info('2D Slice', await rectReader.getSliceND('big-matrix', [[32, 48, 4],[64, 96, 4]]));

    // use yourdataset.getSliceNDArray() to get an NDArray instead of a JS Object.
    ```

=== "C++"

    ```cpp
    #include <stdio.h>
    #include "stards.h"
    using namespace star;

    auto ds = StarDataset::create("data.stards");
    ds->put("big_matrix", NDArray<double>::zeros({10000, 10000}));
    ds->flush();
    ds->close();

    auto sliceReader = StarDataset::open("data.stards", "r");
    // Slice specs are per-dimension {start, stop[, step]}
    auto subset = sliceReader->get_slice<double>("big_matrix", {{0, 100}, {0, 100}});
    auto shape = subset.shape();
    printf("Shape: ");
    for(size_t i = 0; i < shape.size(); i++){
        printf("%zu ", shape[i]);
    }
    printf("\n");
    ```

### 1D, 2D, and 3D examples

=== "Python"

    ```python
    # 1D slice [1000:2000]
    subset = store.get_slice("large_array", [(1000, 2000)])
    print(subset.shape)  # (1000,)

    # 2D slice [10:20, 30:40]
    subset_2d = store.get_slice("big_matrix", [(10, 20), (30, 40)])
    print(subset_2d.shape)  # (10, 10)

    # 3D slice with a step in the first dimension
    subset_3d = store.get_slice("3d_data", [(0, 20, 2), (5, 15), (0, 20)])
    ```

=== "JS"

    ```js
    // 1D Slice: getSlice(key, start, count)
    // returns flat typed array. 1D only.
    const subset = ds.getSlice("large_array", 1000, 1000);
    console.log(subset.length);

    // Alt 1D Slice: getSliceND(key, [[start, end]])
    // ds.getSliceND("big_matrix", [[1000, 2000]]);
    // returns {data: [], shape: []}

    // 2D Slice from (33, 34) to (43, 44).
    // getSliceND(key, [[1stDimStart, end], [2ndDimStart, end]])
    // returns {data: [], shape: []}
    const subset2D = ds.getSliceND("big_matrix", [[33, 43], [34, 44]]);

    // 3D Slice
    // getSliceND(key, [[1stDimStart, end, step], [2ndDimStart, end], [3rdDimStart, end]])
    // returns {data: [], shape: []}
    const subset3D = ds.getSliceND("3d_data", [[0, 20, 2], [5, 15], [0, 20]]);

    // use getSliceNDArray() to return an NDArray
    // instead of a plain {data:[], shape:[]} object.
    ```

## Checking whether an array is sliceable

Every value in the **array namespace** (`ds["key"] = …` / `store.put(...)`) is
stored as its own array and is sliceable regardless of size. Only values written
to the **metadata namespace** (`ds.meta["key"] = …`) live in the shared
[metadata block](compression.md#the-metadata-block) and **cannot** be sliced.

`is_sliceable()` reports this — check first if you're unsure which namespace a key
came from:

=== "Python"

    ```python
    if store.is_sliceable("large_array"):
        subset = store.get_slice("large_array", [(1000, 2000)])
    ```

So to keep something sliceable, simply store it as an array (the default) rather
than as metadata.

## Notes and limits

- Slicing currently supports **1D, 2D, and 3D** arrays.
- Arrays are stored in **row-major (C-style)** order.
- A full runnable example ships at `bindings/python/examples/numpy_interop.py`.
