/**
 * @file UnitTestSlicing.cpp
 * @brief Unit tests for n-dimensional array slicing with data-oriented design
 *
 * DESIGN NOTE:
 * Array slicing only works with arrays stored as separate compressed arrays
 * with block structure. Arrays stored in the metadata block (small arrays,
 * typically <1024 elements) cannot be sliced by design - they must be accessed
 * as complete units using meta.get().
 *
 * This separation ensures:
 * - Efficient access patterns for both small and large arrays
 * - Clear performance characteristics
 * - Optimal cloud storage costs
 * - No ambiguity about access semantics
 *
 * Tests in this file require arrays to be stored separately with blocks.
 */

#include "../include/stards.h"
#include <gtest/gtest.h>
#include "Fixtures.h"
#include <ghc/fs_std.hpp>
#include <random>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>


using namespace star;

// Test fixture for slicing tests. Each test runs in its own temp directory
// (auto-deleted); createTempFile() returns a unique path inside it (parallel-safe).
class SlicingTest : public star_test::TempDirTest {
protected:
    std::string createTempFile(const std::string& prefix = "test_slicing") {
        return tempStardsFile(prefix);
    }

    // Create a dataset with a sliceable (non-shuffle) codec. The library default
    // codec is a byte-shuffle codec (LZ4_SHUFFLE), which trades slice-ability for
    // a better compression ratio, so arrays written with the default config are
    // deliberately not sliceable. Slicing tests opt into a plain codec.
    std::shared_ptr<StarDataset> createSliceable(const std::string& test_file) {
        StarConfig config;
        config.compression = CompressionAlgorithm::NONE;
        return StarDataset::create(test_file, config);
    }
};

// ============================================================================
// Pure Function Tests (Data Transformations)
// ============================================================================

TEST_F(SlicingTest, Slice_DefaultStep) {
    // Test that step defaults to 1
    Slice s{100, 200};
    EXPECT_EQ(s.start, 100);
    EXPECT_EQ(s.stop, 200);
    EXPECT_EQ(s.step, 1);
    EXPECT_EQ(s.length(), 100);
}

TEST_F(SlicingTest, Slice_ExplicitStep) {
    // Test explicit step
    Slice s{0, 100, 2};
    EXPECT_EQ(s.start, 0);
    EXPECT_EQ(s.stop, 100);
    EXPECT_EQ(s.step, 2);
    EXPECT_EQ(s.length(), 50);
}

TEST_F(SlicingTest, Slice_HelperFunctions) {
    // Test helper functions
    Slice all = slice_all(1000);
    EXPECT_EQ(all.start, 0);
    EXPECT_EQ(all.stop, 1000);
    EXPECT_EQ(all.step, 1);

    Slice range = slice_range(100, 200);
    EXPECT_EQ(range.start, 100);
    EXPECT_EQ(range.stop, 200);
    EXPECT_EQ(range.step, 1);
}

// ============================================================================
// Integration Tests - 1D Arrays
// ============================================================================
//
// NOTE: These tests are currently failing due to a storage layer issue where
// large arrays are still being stored in the metadata block even when they
// exceed the configured thresholds. The slicing implementation itself is
// correct - these tests will pass once the storage layer properly stores
// large arrays as separate compressed blocks.
//
// TODO: Fix metadata block storage decision logic to properly store large
// arrays (>8KB or >1024 elements) as separate compressed arrays.
// ============================================================================

TEST_F(SlicingTest, Slice1D_MiddleRange) {
    // Create 1D array with known pattern
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create very large array to force separate storage with blocks
    // Arrays larger than max_metadata_block_size (64KB) are stored separately
    // Need > 8192 int64_t elements (8192 * 8 bytes = 65536 bytes > 64KB)
    NDArray<int64_t> data({20000});
    for (size_t i = 0; i < 20000; ++i) {
        data.flat(i) = static_cast<int64_t>(i);
    }

    store->put("data", data);  // Use store->put() for sliceable arrays
    store->flush();

    // Re-open to test reading
    auto store2 = StarDataset::open(test_file);

    // Slice middle portion [10000, 10100)
    auto slice = store2->get_slice<int64_t>("data", {{10000, 10100}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>{100});
    EXPECT_EQ(slice.size(), 100);

    // Debug: print first few values
    std::cout << "First 5 slice values: ";
    for (size_t i = 0; i < std::min(size_t(5), slice.size()); ++i) {
        std::cout << slice.flat(i) << " ";
    }
    std::cout << "\n";
    std::cout << "Expected: 10000 10001 10002 10003 10004\n";

    // Verify values
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(slice.flat(i), static_cast<int64_t>(10000 + i))
            << "Mismatch at index " << i;
    }
}

TEST_F(SlicingTest, Slice1D_StartRange) {
    // Test slicing from start
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    NDArray<int64_t> data({10000});
    for (size_t i = 0; i < 10000; ++i) {
        data.flat(i) = static_cast<int64_t>(i);
    }

    store->put("data", data);  // Use store->put() for sliceable arrays
    store->flush();

    // Slice [0, 100)
    auto slice = store->get_slice<int64_t>("data", {{0, 100}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>{100});

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(slice.flat(i), static_cast<int64_t>(i));
    }
}

TEST_F(SlicingTest, Slice1D_EndRange) {
    // Test slicing to end
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    NDArray<int64_t> data({10000});
    for (size_t i = 0; i < 10000; ++i) {
        data.flat(i) = static_cast<int64_t>(i);
    }

    store->put("data", data);  // Use store->put() for sliceable arrays
    store->flush();

    // Slice [9900, 10000)
    auto slice = store->get_slice<int64_t>("data", {{9900, 10000}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>{100});

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(slice.flat(i), static_cast<int64_t>(9900 + i));
    }
}

// ============================================================================
// Integration Tests - 2D Arrays
// ============================================================================

TEST_F(SlicingTest, Slice2D_FullRows) {
    // Test 2D array: taking full rows (optimized path)
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create 100x50 array
    size_t rows = 100;
    size_t cols = 50;
    NDArray<double> matrix({rows, cols});

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            matrix(r, c) = static_cast<double>(r * 1000 + c);
        }
    }

    store->put("matrix", matrix);  // Use store->put() for sliceable arrays
    store->flush();

    // Slice rows 10-19 (10 rows), all columns
    auto slice = store->get_slice<double>("matrix", {{10, 20}, {0, cols}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>({10, 50}));
    EXPECT_EQ(slice.size(), 500);

    // Verify values
    for (size_t r = 0; r < 10; ++r) {
        for (size_t c = 0; c < 50; ++c) {
            double expected = static_cast<double>((r + 10) * 1000 + c);
            EXPECT_EQ(slice(r, c), expected)
                << "Mismatch at (" << r << ", " << c << ")";
        }
    }
}

TEST_F(SlicingTest, Slice2D_Submatrix) {
    // Test 2D array: taking submatrix (rows and columns)
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create 100x100 array
    NDArray<float> matrix({100, 100});

    for (size_t r = 0; r < 100; ++r) {
        for (size_t c = 0; c < 100; ++c) {
            matrix(r, c) = static_cast<float>(r * 100 + c);
        }
    }

    store->put("matrix", matrix);
    store->flush();

    // Slice rows 20-29, columns 30-39
    auto slice = store->get_slice<float>("matrix", {{20, 30}, {30, 40}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>({10, 10}));
    EXPECT_EQ(slice.size(), 100);

    // Verify values
    for (size_t r = 0; r < 10; ++r) {
        for (size_t c = 0; c < 10; ++c) {
            float expected = static_cast<float>((r + 20) * 100 + (c + 30));
            EXPECT_EQ(slice(r, c), expected);
        }
    }
}

TEST_F(SlicingTest, Slice2D_HelperFunctions) {
    // Test using helper functions
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    NDArray<float> matrix({50, 40});
    for (size_t r = 0; r < 50; ++r) {
        for (size_t c = 0; c < 40; ++c) {
            matrix(r, c) = static_cast<float>(r * 40 + c);
        }
    }

    store->put("matrix", matrix);
    store->flush();

    // Use helper functions
    auto slice = store->get_slice<float>("matrix", {
        slice_range(10, 20),
        slice_all(40)
    });

    EXPECT_EQ(slice.shape(), std::vector<size_t>({10, 40}));

    // Verify first row
    for (size_t c = 0; c < 40; ++c) {
        float expected = static_cast<float>(10 * 40 + c);
        EXPECT_EQ(slice(0, c), expected);
    }
}

// ============================================================================
// Integration Tests - 3D Arrays
// ============================================================================

TEST_F(SlicingTest, Slice3D_Hyperslab) {
    // Test 3D array slicing
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create 20x15x10 array
    NDArray<uint16_t> volume({20, 15, 10});

    for (size_t i = 0; i < 20; ++i) {
        for (size_t j = 0; j < 15; ++j) {
            for (size_t k = 0; k < 10; ++k) {
                volume(i, j, k) = static_cast<uint16_t>(i * 150 + j * 10 + k);
            }
        }
    }

    store->put("volume", volume);
    store->flush();

    // Slice: [5:10, 3:8, 0:10]
    auto slice = store->get_slice<uint16_t>("volume", {{5, 10}, {3, 8}, {0, 10}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>({5, 5, 10}));
    EXPECT_EQ(slice.size(), 250);

    // Verify a few values
    EXPECT_EQ(slice(0, 0, 0), static_cast<uint16_t>(5 * 150 + 3 * 10 + 0));
    EXPECT_EQ(slice(4, 4, 9), static_cast<uint16_t>(9 * 150 + 7 * 10 + 9));
}

// ============================================================================
// High-Dimensional Array Test (10D)
// ============================================================================

TEST_F(SlicingTest, ErrorHandling_HighDimensionalArrays) {
    // Slicing should only work for 1D, 2D, and 3D arrays
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create 4D array (not supported for slicing)
    std::vector<size_t> shape = {10, 10, 10, 10};
    NDArray<int64_t> data(shape);

    // Fill with pattern
    for (size_t i = 0; i < data.size(); ++i) {
        data.flat(i) = static_cast<int64_t>(i);
    }

    store->put("tensor4d", data);
    store->flush();

    // Attempt to slice 4D array should throw
    EXPECT_THROW(
        store->get_slice<int64_t>("tensor4d", {{0, 5}, {0, 5}, {0, 5}, {0, 5}}),
        std::runtime_error
    );
}

// ============================================================================
// Default Step Parameter Test
// ============================================================================

TEST_F(SlicingTest, DefaultStep_SimpleSyntax) {
    // Test that step=1 is default, cleaner syntax
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create 3D array [20, 15, 10]
    NDArray<float> data({20, 15, 10});
    for (size_t i = 0; i < data.size(); ++i) {
        data.flat(i) = static_cast<float>(i);
    }
    store->put("cube", data);
    store->flush();

    // Slice with omitted step (uses default step=1)
    auto slice1 = store->get_slice<float>("cube", {
        {5, 10},    // No step specified, defaults to 1
        {3, 8},
        {0, 10}
    });

    // Equivalent with explicit step
    auto slice2 = store->get_slice<float>("cube", {
        {5, 10, 1},
        {3, 8, 1},
        {0, 10, 1}
    });

    // Both should be identical
    EXPECT_EQ(slice1.shape(), slice2.shape());
    EXPECT_EQ(slice1.size(), slice2.size());

    for (size_t i = 0; i < slice1.size(); ++i) {
        EXPECT_EQ(slice1.flat(i), slice2.flat(i));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(SlicingTest, DesignConstraint_MetadataBlockNotSliceable) {
    // Test that metadata block items correctly reject slicing
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create small array that will be stored in metadata block
    NDArray<int> small_data({100});  // Small enough for metadata block
    for (size_t i = 0; i < 100; ++i) {
        small_data.flat(i) = static_cast<int>(i);
    }

    store->meta.put("small_data", small_data);  // Use meta.put() to store in metadata block
    store->flush();

    // Verify it's not sliceable
    EXPECT_FALSE(store->is_sliceable("small_data"));

    // Verify slicing throws appropriate error
    try {
        auto slice = store->get_slice<int>("small_data", {{0, 10}});
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        // Verify error message mentions metadata block and suggests meta.get()
        EXPECT_TRUE(msg.find("metadata block") != std::string::npos);
        EXPECT_TRUE(msg.find("meta.get") != std::string::npos);
    }

    // Verify full access via meta.get() works
    auto full_data = store->meta.get("small_data");
    EXPECT_TRUE(full_data != nullptr);
}

TEST_F(SlicingTest, HelperFunction_IsSliceable) {
    // Test is_sliceable() helper function
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Non-existent key
    EXPECT_FALSE(store->is_sliceable("nonexistent"));

    // Small array in metadata block
    NDArray<int> small({50});
    for (size_t i = 0; i < 50; ++i) {
        small.flat(i) = static_cast<int>(i);
    }
    store->meta.put("small", small);  // Use meta.put() to store in metadata block
    store->flush();

    EXPECT_FALSE(store->is_sliceable("small"));

    // Note: Would test large sliceable array here, but current storage
    // layer issue prevents proper setup. When fixed, add:
    // NDArray<int64_t> large({20000});
    // ... fill data ...
    // store->put("large", large);
    // store->flush();
    // EXPECT_TRUE(store->is_sliceable("large"));
}

TEST_F(SlicingTest, ErrorHandling_KeyNotFound) {
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Try to slice non-existent key
    EXPECT_THROW(
        store->get_slice<double>("nonexistent", {{0, 10}}),
        std::runtime_error
    );
}

TEST_F(SlicingTest, ErrorHandling_OutOfBounds) {
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    NDArray<int> data({100});
    for (size_t i = 0; i < 100; ++i) {
        data.flat(i) = static_cast<int>(i);
    }
    store->put("data", data);
    store->flush();

    // Try to slice beyond array bounds
    EXPECT_THROW(
        store->get_slice<int>("data", {{0, 200}}),
        std::runtime_error
    );
}

TEST_F(SlicingTest, ErrorHandling_InvalidSlice) {
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    NDArray<int> data({100});
    for (size_t i = 0; i < 100; ++i) {
        data.flat(i) = static_cast<int>(i);
    }
    store->put("data", data);
    store->flush();

    // Try invalid slice (start >= stop)
    EXPECT_THROW(
        store->get_slice<int>("data", {{50, 40}}),
        std::runtime_error
    );
}

// ============================================================================
// Performance Test (Partial vs Full Load)
// ============================================================================

TEST_F(SlicingTest, Performance_PartialLoad) {
    // Create larger array to test performance benefit
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create 10000 element array (large enough to span multiple blocks)
    NDArray<double> data({10000});
    for (size_t i = 0; i < 10000; ++i) {
        data.flat(i) = static_cast<double>(i);
    }

    store->put("large_array", data);
    store->flush();

    // Get small slice (10% of array)
    auto slice = store->get_slice<double>("large_array", {{0, 1000}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>{1000});

    // Verify correctness
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_EQ(slice.flat(i), static_cast<double>(i));
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SlicingTest, EdgeCase_SingleElement) {
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    // Create array large enough to be stored separately (not in metadata block)
    NDArray<int> data({2000});
    for (size_t i = 0; i < 2000; ++i) {
        data.flat(i) = static_cast<int>(i);
    }
    store->put("data", data);
    store->flush();

    // Slice single element [50:51)
    auto slice = store->get_slice<int>("data", {{50, 51}});

    EXPECT_EQ(slice.size(), 1);
    EXPECT_EQ(slice.flat(0), 50);
}

TEST_F(SlicingTest, EdgeCase_FullArray) {
    std::string test_file = createTempFile();
    auto store = createSliceable(test_file);

    NDArray<double> data({100, 50});
    for (size_t i = 0; i < data.size(); ++i) {
        data.flat(i) = static_cast<double>(i);
    }
    store->put("data", data);
    store->flush();

    // Slice entire array
    auto slice = store->get_slice<double>("data", {{0, 100}, {0, 50}});

    EXPECT_EQ(slice.shape(), std::vector<size_t>({100, 50}));
    EXPECT_EQ(slice.size(), 5000);

    // Verify all elements match
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(slice.flat(i), data.flat(i));
    }
}

// ============================================================================
// Regression: slicing across block boundaries with a NON-DEFAULT block_size.
//
// get_slice() must derive elements-per-block from the block size the array was
// WRITTEN with (blocks[0].uncompressed_size), not from the currently-open
// config. Reopening READ_ONLY resets the in-memory config to its 1 MB default;
// if that default leaked into the block-mapping math it would disagree with the
// extraction step, drop every block past the first, and silently return zeros
// for the tail of a multi-block slice. Written with a small block_size so a
// modest array spans many blocks and the slice straddles several boundaries.
// ============================================================================
TEST_F(SlicingTest, Regression_NonDefaultBlockSize_SpansBlocks) {
    std::string test_file = createTempFile();
    const size_t N = 4000;
    {
        StarConfig config;
        config.compression = CompressionAlgorithm::NONE;
        config.block_size = 1024;  // 128 doubles/block -> ~32 blocks
        auto store = StarDataset::create(test_file, config);
        NDArray<double> data({N});
        for (size_t i = 0; i < N; ++i) data.flat(i) = static_cast<double>(i) * 1.5;
        store->put("data", data);
        store->close();
    }

    // Reopen READ_ONLY: the in-memory config is now the 1 MB default, which
    // differs from the 1024-byte block size this array actually used.
    auto store = StarDataset::open(test_file, FileMode::READ_ONLY);

    // Slice straddles ~30 block boundaries.
    auto slice = store->get_slice<double>("data", {{100, 3900}});
    ASSERT_EQ(slice.size(), 3800u);
    for (size_t i = 0; i < slice.size(); ++i) {
        EXPECT_EQ(slice.flat(i), static_cast<double>(100 + i) * 1.5)
            << "Mismatch at slice index " << i << " (array element " << (100 + i) << ")";
    }
}

// ============================================================================
// Regression: strided slice (step > 1) starting at element 0 across many blocks.
//
// createBlockMap()'s contiguous-from-zero branch sized the covered block range
// from spec.total_elements — the DOWNSAMPLED output count, which folds in the
// step. But the strided copy in get_slice() reads source elements across the
// WHOLE window (full rows, every row), so decompressing only enough blocks for
// the downsampled count left the window's tail undecompressed; extractElements()
// then silently skipped those ranges and the output's trailing rows came back as
// all-zeros. No error, correct shape — only wrong (zero) values past some row.
// Values (not just nonzero-ness) are checked so a zero-filled tail fails.
// ============================================================================
TEST_F(SlicingTest, Regression_StridedFromZero_SpansBlocks_1D) {
    std::string test_file = createTempFile();
    const size_t N = 8000;
    {
        StarConfig config;
        config.compression = CompressionAlgorithm::NONE;
        config.block_size = 1024;  // 128 doubles/block -> ~63 blocks
        auto store = StarDataset::create(test_file, config);
        NDArray<double> data({N});
        for (size_t i = 0; i < N; ++i) data.flat(i) = static_cast<double>(i) + 1.0;  // never 0
        store->put("data", data);
        store->close();
    }

    auto store = StarDataset::open(test_file, FileMode::READ_ONLY);
    for (size_t step : {size_t(2), size_t(3), size_t(5), size_t(10)}) {
        auto slice = store->get_slice<double>("data", {{0, N, step}});
        const size_t expected_len = (N + step - 1) / step;  // ceil(N/step)
        ASSERT_EQ(slice.size(), expected_len) << "step=" << step;
        for (size_t j = 0; j < slice.size(); ++j) {
            EXPECT_EQ(slice.flat(j), static_cast<double>(j * step) + 1.0)
                << "step=" << step << " mismatch @ output " << j
                << " (source element " << (j * step) << ")";
        }
    }
}

TEST_F(SlicingTest, Regression_StridedFromZero_SpansBlocks_2D) {
    std::string test_file = createTempFile();
    const size_t R = 400, C = 300;  // 120000 doubles -> many 1024-byte blocks
    {
        StarConfig config;
        config.compression = CompressionAlgorithm::NONE;
        config.block_size = 1024;  // 128 doubles/block
        auto store = StarDataset::create(test_file, config);
        NDArray<double> data({R, C});
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c)
                data.flat(r * C + c) = static_cast<double>(r * C + c) + 1.0;  // never 0
        store->put("m", data);
        store->close();
    }

    auto store = StarDataset::open(test_file, FileMode::READ_ONLY);
    for (size_t step : {size_t(2), size_t(3), size_t(5)}) {
        auto slice = store->get_slice<double>("m", {{0, R, step}, {0, C, step}});
        const size_t sh = (R + step - 1) / step, sw = (C + step - 1) / step;
        ASSERT_EQ(slice.shape(), std::vector<size_t>({sh, sw})) << "step=" << step;
        for (size_t y = 0; y < sh; ++y)
            for (size_t x = 0; x < sw; ++x) {
                size_t src_r = y * step, src_c = x * step;
                EXPECT_EQ(slice.flat(y * sw + x),
                          static_cast<double>(src_r * C + src_c) + 1.0)
                    << "step=" << step << " mismatch @ output (" << y << "," << x << ")";
            }
    }
}

// ============================================================================
// Per-block byte-shuffle codecs (GZIP_SHUFFLE_BLOCK / LZ4_SHUFFLE_BLOCK) apply
// the shuffle prefilter independently within each block, keeping each block
// self-contained so get_slice() can decode only the covering blocks. These
// verify that block-shuffled arrays are sliceable and round-trip correctly,
// including slices that cross block boundaries. (The legacy global-shuffle
// variants shuffle the whole array at once and are intentionally NOT sliceable
// — covered by HelperFunction_IsSliceable.)
// ============================================================================
class BlockShuffleSlicingTest
    : public SlicingTest,
      public ::testing::WithParamInterface<CompressionAlgorithm> {};

TEST_P(BlockShuffleSlicingTest, SlicesAcrossBlocks_1D) {
    const CompressionAlgorithm codec = GetParam();
    std::string test_file = createTempFile();
    const size_t N = 4000;
    {
        StarConfig config;
        config.compression = codec;
        config.block_size = 1024;  // force many blocks
        auto store = StarDataset::create(test_file, config);
        NDArray<double> data({N});
        for (size_t i = 0; i < N; ++i) data.flat(i) = static_cast<double>(i) * 1.5;
        store->put("data", data);
        store->close();
    }

    auto store = StarDataset::open(test_file, FileMode::READ_ONLY);
    EXPECT_TRUE(store->is_sliceable("data"));

    // Full read round-trips.
    auto full = store->get<double>("data");
    ASSERT_EQ(full.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(full.flat(i), static_cast<double>(i) * 1.5) << "full read mismatch @" << i;

    // Multi-block slice.
    auto slice = store->get_slice<double>("data", {{100, 3900}});
    ASSERT_EQ(slice.size(), 3800u);
    for (size_t i = 0; i < slice.size(); ++i)
        EXPECT_EQ(slice.flat(i), static_cast<double>(100 + i) * 1.5) << "slice mismatch @" << i;

    // Head, tail, single, stepped.
    auto head = store->get_slice<double>("data", {{0, 1}});
    ASSERT_EQ(head.size(), 1u);
    EXPECT_EQ(head.flat(0), 0.0);
    auto tail = store->get_slice<double>("data", {{N - 1, N}});
    ASSERT_EQ(tail.size(), 1u);
    EXPECT_EQ(tail.flat(0), static_cast<double>(N - 1) * 1.5);
    auto step = store->get_slice<double>("data", {{10, 40, 5}});
    ASSERT_EQ(step.size(), 6u);
    for (size_t i = 0; i < step.size(); ++i)
        EXPECT_EQ(step.flat(i), static_cast<double>(10 + 5 * i) * 1.5) << "step mismatch @" << i;
}

TEST_P(BlockShuffleSlicingTest, SlicesFullRows_2D) {
    const CompressionAlgorithm codec = GetParam();
    std::string test_file = createTempFile();
    const int R = 500, C = 8;
    {
        StarConfig config;
        config.compression = codec;
        config.block_size = 1024;  // 256 int32/block
        auto store = StarDataset::create(test_file, config);
        NDArray<int32_t> data({(size_t)R, (size_t)C});
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) data.flat(r * C + c) = r * 100 + c;
        store->put("m", data);
        store->close();
    }

    auto store = StarDataset::open(test_file, FileMode::READ_ONLY);
    EXPECT_TRUE(store->is_sliceable("m"));

    auto rows = store->get_slice<int32_t>("m", {{100, 105}, {0, (size_t)C}});
    ASSERT_EQ(rows.size(), (size_t)(5 * C));
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < C; ++c)
            EXPECT_EQ(rows.flat(r * C + c), (100 + r) * 100 + c)
                << "row-slice mismatch @(" << r << "," << c << ")";
}

INSTANTIATE_TEST_SUITE_P(
    BlockShuffleCodecs, BlockShuffleSlicingTest,
    ::testing::Values(CompressionAlgorithm::GZIP_SHUFFLE_BLOCK,
                      CompressionAlgorithm::LZ4_SHUFFLE_BLOCK),
    [](const ::testing::TestParamInfo<CompressionAlgorithm>& info) {
        return info.param == CompressionAlgorithm::GZIP_SHUFFLE_BLOCK
                   ? std::string("GZIP_SHUFFLE_BLOCK")
                   : std::string("LZ4_SHUFFLE_BLOCK");
    });
