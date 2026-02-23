#include <cub/device/device_segmented_radix_sort.cuh>

#include <cuda_runtime.h>

#include "../nb_types.h"
#include "kernels_wilcoxon.cuh"

using namespace nb::literals;

// Constants for kernel launch configuration
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

// Query CUB temp storage requirement for segmented radix sort
static size_t get_seg_sort_temp_bytes(int n_rows, int n_cols) {
    size_t bytes = 0;
    auto* dk = reinterpret_cast<double*>(1);
    auto* dv = reinterpret_cast<int*>(1);
    auto* doff = reinterpret_cast<int*>(1);
    int n_items = n_rows * n_cols;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, bytes, dk, dk, dv, dv, n_items, n_cols, doff, doff + 1, 0, 64);
    return bytes;
}

/**
 * Fused compute_ranks: CUB segmented radix sort + average rank + tie
 * correction.
 *
 * Replaces the Python-side argsort + take_along_axis + asfortranarray pipeline
 * with a single C++ call. All buffers are allocated by the caller (Python/CuPy)
 * so they go through the same memory pool (RMM-compatible).
 */
static inline void compute_ranks_impl(
    double* matrix,       // F-order (n_rows, n_cols), overwritten with ranks
    double* correction,   // (n_cols) output for tie correction factors
    double* sorted_vals,  // (n_rows * n_cols) workspace for sorted values
    int* sorter,          // (n_rows * n_cols) workspace for sort indices
    int* iota,            // (n_rows * n_cols) workspace for iota values
    int* offsets,         // (n_cols + 1) segment offsets
    uint8_t* cub_temp,    // CUB temporary storage
    size_t cub_temp_bytes, int n_rows, int n_cols, cudaStream_t stream) {
    if (n_rows == 0 || n_cols == 0) return;

    int n_items = n_rows * n_cols;

    // Initialize iota: each column gets [0, 1, ..., n_rows-1]
    // and fill segment offsets: [0, n_rows, 2*n_rows, ...]
    {
        constexpr int THREADS = 256;
        dim3 iota_grid((n_rows + THREADS - 1) / THREADS, n_cols);
        iota_segments_kernel<<<iota_grid, THREADS, 0, stream>>>(iota, n_rows,
                                                                n_cols);

        int off_blocks = (n_cols + 1 + THREADS - 1) / THREADS;
        fill_offsets_kernel<<<off_blocks, THREADS, 0, stream>>>(offsets, n_rows,
                                                                n_cols);
    }

    // CUB segmented radix sort
    cub::DeviceSegmentedRadixSort::SortPairs(
        cub_temp, cub_temp_bytes, matrix, sorted_vals, iota, sorter, n_items,
        n_cols, offsets, offsets + 1, 0, 64, stream);

    // Compute average ranks (writes back to matrix)
    int threads = round_up_to_warp(n_rows);
    average_rank_kernel<<<n_cols, threads, 0, stream>>>(sorted_vals, sorter,
                                                        matrix, n_rows, n_cols);

    // Compute tie correction factors
    tie_correction_kernel<<<n_cols, threads, 0, stream>>>(
        sorted_vals, correction, n_rows, n_cols);
}

static inline void launch_tie_correction(const double* sorted_vals,
                                         double* correction, int n_rows,
                                         int n_cols, cudaStream_t stream) {
    int threads_per_block = round_up_to_warp(n_rows);
    dim3 block(threads_per_block);
    dim3 grid(n_cols);
    tie_correction_kernel<<<grid, block, 0, stream>>>(sorted_vals, correction,
                                                      n_rows, n_cols);
}

static inline void launch_average_rank(const double* sorted_vals,
                                       const int* sorter, double* ranks,
                                       int n_rows, int n_cols,
                                       cudaStream_t stream) {
    int threads_per_block = round_up_to_warp(n_rows);
    dim3 block(threads_per_block);
    dim3 grid(n_cols);
    average_rank_kernel<<<grid, block, 0, stream>>>(sorted_vals, sorter, ranks,
                                                    n_rows, n_cols);
}

NB_MODULE(_wilcoxon_cuda, m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test";

    // Query CUB temp storage size for segmented sort
    m.def("get_sort_temp_bytes", &get_seg_sort_temp_bytes, "n_rows"_a,
          "n_cols"_a);

    // Fused compute_ranks: CUB segmented sort + average rank + tie correction
    m.def(
        "compute_ranks",
        [](cuda_array_f<double> matrix, cuda_array<double> correction,
           cuda_array_f<double> sorted_vals, cuda_array_f<int> sorter,
           cuda_array_f<int> iota, cuda_array<int> offsets,
           cuda_array<uint8_t> cub_temp, int n_rows, int n_cols,
           std::uintptr_t stream) {
            compute_ranks_impl(matrix.data(), correction.data(),
                               sorted_vals.data(), sorter.data(), iota.data(),
                               offsets.data(), cub_temp.data(), cub_temp.size(),
                               n_rows, n_cols, (cudaStream_t)stream);
        },
        "matrix"_a, "correction"_a, "sorted_vals"_a, "sorter"_a, "iota"_a,
        "offsets"_a, "cub_temp"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);

    // Tie correction kernel
    m.def(
        "tie_correction",
        [](cuda_array_f<const double> sorted_vals,
           cuda_array<double> correction, int n_rows, int n_cols,
           std::uintptr_t stream) {
            launch_tie_correction(sorted_vals.data(), correction.data(), n_rows,
                                  n_cols, (cudaStream_t)stream);
        },
        "sorted_vals"_a, "correction"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);

    // Average rank kernel
    m.def(
        "average_rank",
        [](cuda_array_f<const double> sorted_vals,
           cuda_array_f<const int> sorter, cuda_array_f<double> ranks,
           int n_rows, int n_cols, std::uintptr_t stream) {
            launch_average_rank(sorted_vals.data(), sorter.data(), ranks.data(),
                                n_rows, n_cols, (cudaStream_t)stream);
        },
        "sorted_vals"_a, "sorter"_a, "ranks"_a, nb::kw_only(), "n_rows"_a,
        "n_cols"_a, "stream"_a = 0);
}
