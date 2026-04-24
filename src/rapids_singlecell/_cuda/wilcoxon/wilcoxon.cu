#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_wilcoxon.cuh"

using namespace nb::literals;

// Constants for kernel launch configuration
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int OVO_THREADS_PER_BLOCK = 256;

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

static inline void launch_ovo_rank_dense(
    const float* ref_sorted, const float* grp_data, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, cudaStream_t stream) {
    dim3 block(OVO_THREADS_PER_BLOCK);
    dim3 grid(n_cols, n_groups);
    ovo_rank_dense_kernel<<<grid, block, 0, stream>>>(
        ref_sorted, grp_data, grp_offsets, rank_sums, tie_corr, n_ref,
        n_all_grp, n_cols, n_groups, compute_tie_corr);
    CUDA_CHECK_LAST_ERROR(ovo_rank_dense_kernel);
}

static inline void launch_ovo_rank_presorted(
    const float* ref_sorted, const float* grp_sorted, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, cudaStream_t stream) {
    dim3 block(OVO_THREADS_PER_BLOCK);
    dim3 grid(n_cols, n_groups);
    ovo_rank_presorted_kernel<<<grid, block, 0, stream>>>(
        ref_sorted, grp_sorted, grp_offsets, rank_sums, tie_corr, n_ref,
        n_all_grp, n_cols, n_groups, compute_tie_corr);
    CUDA_CHECK_LAST_ERROR(ovo_rank_presorted_kernel);
}

static inline void launch_ovr_rank_dense(
    const float* sorted_vals, const int* sorter, const int* group_codes,
    double* rank_sums, double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, cudaStream_t stream) {
    int threads_per_block = round_up_to_warp(n_rows);
    dim3 block(threads_per_block);
    dim3 grid(n_cols);
    ovr_rank_dense_kernel<<<grid, block, 0, stream>>>(
        sorted_vals, sorter, group_codes, rank_sums, tie_corr, n_rows, n_cols,
        n_groups, compute_tie_corr);
    CUDA_CHECK_LAST_ERROR(ovr_rank_dense_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test";

    m.def(
        "ovo_rank_dense",
        [](gpu_array_f<const float, Device> ref_sorted,
           gpu_array_f<const float, Device> grp_data,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,
           int n_cols, int n_groups, bool compute_tie_corr,
           std::uintptr_t stream) {
            launch_ovo_rank_dense(
                ref_sorted.data(), grp_data.data(), grp_offsets.data(),
                rank_sums.data(), tie_corr.data(), n_ref, n_all_grp, n_cols,
                n_groups, compute_tie_corr, (cudaStream_t)stream);
        },
        "ref_sorted"_a, "grp_data"_a, "grp_offsets"_a, "rank_sums"_a,
        "tie_corr"_a, nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a, "stream"_a = 0);

    m.def(
        "ovo_rank_presorted",
        [](gpu_array_f<const float, Device> ref_sorted,
           gpu_array_f<const float, Device> grp_sorted,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,
           int n_cols, int n_groups, bool compute_tie_corr,
           std::uintptr_t stream) {
            launch_ovo_rank_presorted(
                ref_sorted.data(), grp_sorted.data(), grp_offsets.data(),
                rank_sums.data(), tie_corr.data(), n_ref, n_all_grp, n_cols,
                n_groups, compute_tie_corr, (cudaStream_t)stream);
        },
        "ref_sorted"_a, "grp_sorted"_a, "grp_offsets"_a, "rank_sums"_a,
        "tie_corr"_a, nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a, "stream"_a = 0);

    m.def(
        "ovr_rank_dense",
        [](gpu_array_f<const float, Device> sorted_vals,
           gpu_array_f<const int, Device> sorter,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, std::uintptr_t stream) {
            launch_ovr_rank_dense(sorted_vals.data(), sorter.data(),
                                  group_codes.data(), rank_sums.data(),
                                  tie_corr.data(), n_rows, n_cols, n_groups,
                                  compute_tie_corr, (cudaStream_t)stream);
        },
        "sorted_vals"_a, "sorter"_a, "group_codes"_a, "rank_sums"_a,
        "tie_corr"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a, "n_groups"_a,
        "compute_tie_corr"_a, "stream"_a = 0);
}

NB_MODULE(_wilcoxon_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
