#pragma once

// ============================================================================
// Nanobind module
// ============================================================================

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test (OVR)";

    // ---- Streaming pipelines ----

    m.def(
        "ovr_streaming",
        [](gpu_array_f<const float, Device> block,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, int sub_batch_cols) {
            ovr_streaming_impl(block.data(), group_codes.data(),
                               rank_sums.data(), tie_corr.data(), n_rows,
                               n_cols, n_groups, compute_tie_corr,
                               sub_batch_cols);
        },
        "block"_a, "group_codes"_a, "rank_sums"_a, "tie_corr"_a, nb::kw_only(),
        "n_rows"_a, "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovr_sparse_csr",
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

    m.def(
        "ovr_sparse_csc",
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

    // ---- Host-streaming pipelines (host inputs, device outputs) ----

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

#define RSC_OVR_DENSE_HOST_BINDING(NAME, InT)                                  \
    m.def(                                                                     \
        NAME,                                                                  \
        [](host_array_2d<const InT> h_block,                                   \
           host_array<const int> h_group_codes,                                \
           gpu_array_c<double, Device> d_rank_sums,                            \
           gpu_array_c<double, Device> d_tie_corr,                             \
           gpu_array_c<double, Device> d_group_sums,                           \
           gpu_array_c<double, Device> d_group_sq_sums,                        \
           gpu_array_c<double, Device> d_group_nnz, int n_rows, int n_cols,    \
           int n_groups, bool compute_tie_corr, bool compute_sq_sums,          \
           bool compute_nnz, int sub_batch_cols) {                             \
            ovr_streaming_dense_host_impl<InT>(                                \
                h_block.data(), h_group_codes.data(), d_rank_sums.data(),      \
                d_tie_corr.data(), d_group_sums.data(),                        \
                d_group_sq_sums.data(), d_group_nnz.data(), n_rows, n_cols,    \
                n_groups, compute_tie_corr, compute_sq_sums, compute_nnz,      \
                sub_batch_cols);                                               \
        },                                                                     \
        "h_block"_a, "h_group_codes"_a, "d_rank_sums"_a, "d_tie_corr"_a,       \
        "d_group_sums"_a, "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), \
        "n_rows"_a, "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,            \
        "compute_sq_sums"_a = true, "compute_nnz"_a = true,                    \
        "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVR_DENSE_HOST_BINDING("ovr_streaming_dense_host", float);
    RSC_OVR_DENSE_HOST_BINDING("ovr_streaming_dense_host_f64", double);
#undef RSC_OVR_DENSE_HOST_BINDING
}
