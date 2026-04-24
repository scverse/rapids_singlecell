#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_fast_common.cuh"
#include "wilcoxon_sparse_kernels.cuh"
#include "wilcoxon_ovr_kernels.cuh"
#include "wilcoxon_ovr_sparse.cuh"
#include "kernels_wilcoxon_ovo.cuh"
#include "wilcoxon_ovo_kernels.cuh"
#include "wilcoxon_ovo_device_sparse.cuh"
#include "wilcoxon_ovo_host_sparse.cuh"

using namespace nb::literals;

template <typename Device>
void register_sparse_bindings(nb::module_& m) {
    m.doc() = "Sparse-native host Wilcoxon CUDA kernels";

    m.def(
        "ovr_sparse_csc_device",
        [](gpu_array_c<const float, Device> csc_data,
           gpu_array_c<const int, Device> csc_indices,
           gpu_array_c<const int, Device> csc_indptr,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<const double, Device> group_sizes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, int sub_batch_cols) {
            ovr_sparse_csc_streaming_impl(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                group_codes.data(), group_sizes.data(), rank_sums.data(),
                tie_corr.data(), n_rows, n_cols, n_groups, compute_tie_corr,
                sub_batch_cols);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "group_codes"_a,
        "group_sizes"_a, "rank_sums"_a, "tie_corr"_a, nb::kw_only(), "n_rows"_a,
        "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovr_sparse_csr_device",
        [](gpu_array_c<const float, Device> csr_data,
           gpu_array_c<const int, Device> csr_indices,
           gpu_array_c<const int, Device> csr_indptr,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<const double, Device> group_sizes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, int sub_batch_cols) {
            ovr_sparse_csr_streaming_impl(
                csr_data.data(), csr_indices.data(), csr_indptr.data(),
                group_codes.data(), group_sizes.data(), rank_sums.data(),
                tie_corr.data(), n_rows, n_cols, n_groups, compute_tie_corr,
                sub_batch_cols);
        },
        "csr_data"_a, "csr_indices"_a, "csr_indptr"_a, "group_codes"_a,
        "group_sizes"_a, "rank_sums"_a, "tie_corr"_a, nb::kw_only(), "n_rows"_a,
        "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

#define RSC_OVR_SPARSE_CSC_HOST_BINDING(NAME, InT, IndptrT)                   \
    m.def(                                                                    \
        NAME,                                                                 \
        [](host_array<const InT> h_data, host_array<const int> h_indices,     \
           host_array<const IndptrT> h_indptr,                                \
           host_array<const int> h_group_codes,                               \
           host_array<double> h_group_sizes,                                  \
           gpu_array_c<double, Device> d_rank_sums,                           \
           gpu_array_c<double, Device> d_tie_corr,                            \
           gpu_array_c<double, Device> d_group_sums,                          \
           gpu_array_c<double, Device> d_group_sq_sums,                       \
           gpu_array_c<double, Device> d_group_nnz, int n_rows, int n_cols,   \
           int n_groups, bool compute_tie_corr, bool compute_sq_sums,         \
           bool compute_nnz, int sub_batch_cols) {                            \
            ovr_sparse_csc_host_streaming_impl<InT, IndptrT>(                 \
                h_data.data(), h_indices.data(), h_indptr.data(),             \
                h_group_codes.data(), h_group_sizes.data(),                   \
                d_rank_sums.data(), d_tie_corr.data(), d_group_sums.data(),   \
                d_group_sq_sums.data(), d_group_nnz.data(), n_rows, n_cols,   \
                n_groups, compute_tie_corr, compute_sq_sums, compute_nnz,     \
                sub_batch_cols);                                              \
        },                                                                    \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_group_codes"_a,           \
        "h_group_sizes"_a, "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a, \
        "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), "n_rows"_a,      \
        "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,                       \
        "compute_sq_sums"_a = true, "compute_nnz"_a = true,                   \
        "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host", float, int);
    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host_i64", float, int64_t);
    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host_f64", double, int);
    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host_f64_i64", double,
                                    int64_t);
#undef RSC_OVR_SPARSE_CSC_HOST_BINDING

#define RSC_OVR_SPARSE_CSR_HOST_BINDING(NAME, InT, IndexT, IndptrT)           \
    m.def(                                                                    \
        NAME,                                                                 \
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,  \
           host_array<const IndptrT> h_indptr,                                \
           host_array<const int> h_group_codes,                               \
           host_array<double> h_group_sizes,                                  \
           gpu_array_c<double, Device> d_rank_sums,                           \
           gpu_array_c<double, Device> d_tie_corr,                            \
           gpu_array_c<double, Device> d_group_sums,                          \
           gpu_array_c<double, Device> d_group_sq_sums,                       \
           gpu_array_c<double, Device> d_group_nnz, int n_rows, int n_cols,   \
           int n_groups, bool compute_tie_corr, bool compute_sq_sums,         \
           bool compute_nnz, int sub_batch_cols) {                            \
            ovr_sparse_csr_host_streaming_impl<InT, IndexT, IndptrT>(         \
                h_data.data(), h_indices.data(), h_indptr.data(),             \
                h_group_codes.data(), h_group_sizes.data(),                   \
                d_rank_sums.data(), d_tie_corr.data(), d_group_sums.data(),   \
                d_group_sq_sums.data(), d_group_nnz.data(), n_rows, n_cols,   \
                n_groups, compute_tie_corr, compute_sq_sums, compute_nnz,     \
                sub_batch_cols);                                              \
        },                                                                    \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_group_codes"_a,           \
        "h_group_sizes"_a, "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a, \
        "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), "n_rows"_a,      \
        "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,                       \
        "compute_sq_sums"_a = true, "compute_nnz"_a = true,                   \
        "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host", float, int, int);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_i64", float, int,
                                    int64_t);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_idx64", float, int64_t,
                                    int);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_idx64_i64", float,
                                    int64_t, int64_t);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_f64", double, int,
                                    int);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_f64_i64", double, int,
                                    int64_t);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_f64_idx64", double,
                                    int64_t, int);
    RSC_OVR_SPARSE_CSR_HOST_BINDING("ovr_sparse_csr_host_f64_idx64_i64", double,
                                    int64_t, int64_t);
#undef RSC_OVR_SPARSE_CSR_HOST_BINDING

    m.def(
        "ovo_streaming_csc_device",
        [](gpu_array_c<const float, Device> csc_data,
           gpu_array_c<const int, Device> csc_indices,
           gpu_array_c<const int, Device> csc_indptr,
           gpu_array_c<const int, Device> ref_row_map,
           gpu_array_c<const int, Device> grp_row_map,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,
           int n_cols, int n_groups, bool compute_tie_corr,
           int sub_batch_cols) {
            ovo_streaming_csc_impl(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                ref_row_map.data(), grp_row_map.data(), grp_offsets.data(),
                rank_sums.data(), tie_corr.data(), n_ref, n_all_grp, n_cols,
                n_groups, compute_tie_corr, sub_batch_cols);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "ref_row_map"_a,
        "grp_row_map"_a, "grp_offsets"_a, "rank_sums"_a, "tie_corr"_a,
        nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a, "n_groups"_a,
        "compute_tie_corr"_a, "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovo_streaming_csr_device",
        [](gpu_array_c<const float, Device> csr_data,
           gpu_array_c<const int, Device> csr_indices,
           gpu_array_c<const int, Device> csr_indptr,
           gpu_array_c<const int, Device> ref_row_ids,
           gpu_array_c<const int, Device> grp_row_ids,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,
           int n_cols, int n_groups, bool compute_tie_corr,
           int sub_batch_cols) {
            ovo_streaming_csr_impl(
                csr_data.data(), csr_indices.data(), csr_indptr.data(),
                ref_row_ids.data(), grp_row_ids.data(), grp_offsets.data(),
                rank_sums.data(), tie_corr.data(), n_ref, n_all_grp, n_cols,
                n_groups, compute_tie_corr, sub_batch_cols);
        },
        "csr_data"_a, "csr_indices"_a, "csr_indptr"_a, "ref_row_ids"_a,
        "grp_row_ids"_a, "grp_offsets"_a, "rank_sums"_a, "tie_corr"_a,
        nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a, "n_groups"_a,
        "compute_tie_corr"_a, "sub_batch_cols"_a = SUB_BATCH_COLS);

#define RSC_OVO_CSC_HOST_BINDING(NAME, InT, IndexT, IndptrT)                  \
    m.def(                                                                    \
        NAME,                                                                 \
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,  \
           host_array<const IndptrT> h_indptr,                                \
           host_array<const int> h_ref_row_map,                               \
           host_array<const int> h_grp_row_map,                               \
           host_array<const int> h_grp_offsets,                               \
           host_array<const int> h_stats_codes,                               \
           gpu_array_c<double, Device> d_rank_sums,                           \
           gpu_array_c<double, Device> d_tie_corr,                            \
           gpu_array_c<double, Device> d_group_sums,                          \
           gpu_array_c<double, Device> d_group_sq_sums,                       \
           gpu_array_c<double, Device> d_group_nnz, int n_ref, int n_all_grp, \
           int n_rows, int n_cols, int n_groups, int n_groups_stats,          \
           bool compute_tie_corr, bool compute_sq_sums, bool compute_nnz,     \
           int sub_batch_cols) {                                              \
            ovo_streaming_csc_host_impl<InT, IndexT, IndptrT>(                \
                h_data.data(), h_indices.data(), h_indptr.data(),             \
                h_ref_row_map.data(), h_grp_row_map.data(),                   \
                h_grp_offsets.data(), h_stats_codes.data(),                   \
                d_rank_sums.data(), d_tie_corr.data(), d_group_sums.data(),   \
                d_group_sq_sums.data(), d_group_nnz.data(), n_ref, n_all_grp, \
                n_rows, n_cols, n_groups, n_groups_stats, compute_tie_corr,   \
                compute_sq_sums, compute_nnz, sub_batch_cols);                \
        },                                                                    \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_ref_row_map"_a,           \
        "h_grp_row_map"_a, "h_grp_offsets"_a, "h_stats_codes"_a,              \
        "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a,                    \
        "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), "n_ref"_a,       \
        "n_all_grp"_a, "n_rows"_a, "n_cols"_a, "n_groups"_a,                  \
        "n_groups_stats"_a, "compute_tie_corr"_a, "compute_sq_sums"_a = true, \
        "compute_nnz"_a = true, "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host", float, int, int);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_i64", float, int, int64_t);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_idx64", float, int64_t,
                             int);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_idx64_i64", float, int64_t,
                             int64_t);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_f64", double, int, int);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_f64_i64", double, int,
                             int64_t);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_f64_idx64", double,
                             int64_t, int);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host_f64_idx64_i64", double,
                             int64_t, int64_t);
#undef RSC_OVO_CSC_HOST_BINDING

#define RSC_OVO_CSR_HOST_BINDING(NAME, InT, IndexT, IndptrT)                   \
    m.def(                                                                     \
        NAME,                                                                  \
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,   \
           host_array<const IndptrT> h_indptr,                                 \
           host_array<const int> h_ref_row_ids,                                \
           host_array<const int> h_grp_row_ids,                                \
           host_array<const int> h_grp_offsets,                                \
           gpu_array_c<double, Device> d_rank_sums,                            \
           gpu_array_c<double, Device> d_tie_corr,                             \
           gpu_array_c<double, Device> d_group_sums,                           \
           gpu_array_c<double, Device> d_group_sq_sums,                        \
           gpu_array_c<double, Device> d_group_nnz, int n_full_rows,           \
           int n_ref, int n_all_grp, int n_cols, int n_test,                   \
           int n_groups_stats, bool compute_tie_corr, bool compute_sq_sums,    \
           bool compute_nnz, bool compute_sums, int sub_batch_cols) {          \
            ovo_streaming_csr_host_impl<InT, IndexT, IndptrT>(                 \
                h_data.data(), h_indices.data(), h_indptr.data(), n_full_rows, \
                h_ref_row_ids.data(), n_ref, h_grp_row_ids.data(),             \
                h_grp_offsets.data(), n_all_grp, n_test, d_rank_sums.data(),   \
                d_tie_corr.data(), d_group_sums.data(),                        \
                d_group_sq_sums.data(), d_group_nnz.data(), n_cols,            \
                n_groups_stats, compute_tie_corr, compute_sq_sums,             \
                compute_nnz, compute_sums, sub_batch_cols);                    \
        },                                                                     \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_ref_row_ids"_a,            \
        "h_grp_row_ids"_a, "h_grp_offsets"_a, "d_rank_sums"_a, "d_tie_corr"_a, \
        "d_group_sums"_a, "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), \
        "n_full_rows"_a, "n_ref"_a, "n_all_grp"_a, "n_cols"_a, "n_test"_a,     \
        "n_groups_stats"_a, "compute_tie_corr"_a, "compute_sq_sums"_a = true,  \
        "compute_nnz"_a = true, "compute_sums"_a = true,                       \
        "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host", float, int, int);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_i64", float, int, int64_t);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_idx64", float, int64_t,
                             int);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_idx64_i64", float, int64_t,
                             int64_t);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_f64", double, int, int);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_f64_i64", double, int,
                             int64_t);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_f64_idx64", double,
                             int64_t, int);
    RSC_OVO_CSR_HOST_BINDING("ovo_streaming_csr_host_f64_idx64_i64", double,
                             int64_t, int64_t);
#undef RSC_OVO_CSR_HOST_BINDING
}

NB_MODULE(_wilcoxon_sparse_cuda, m) {
    REGISTER_GPU_BINDINGS(register_sparse_bindings, m);
}
