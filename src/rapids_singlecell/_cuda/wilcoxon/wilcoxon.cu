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
