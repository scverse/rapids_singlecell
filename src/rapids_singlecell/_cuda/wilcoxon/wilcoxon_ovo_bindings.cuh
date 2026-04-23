#pragma once

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test (OVO)";

    // ---- Utility bindings (CUB sort, CSR extraction) ----

    m.def("get_seg_sort_temp_bytes", &get_seg_sort_temp_bytes, "n_items"_a,
          "n_segments"_a);

    m.def(
        "segmented_sort",
        [](gpu_array_c<const float, Device> keys_in,
           gpu_array_c<float, Device> keys_out,
           gpu_array_c<const int, Device> offsets,
           gpu_array_c<uint8_t, Device> cub_temp, int n_items, int n_segments,
           std::uintptr_t stream) {
            size_t temp_bytes = cub_temp.size();
            cub::DeviceSegmentedRadixSort::SortKeys(
                cub_temp.data(), temp_bytes, keys_in.data(), keys_out.data(),
                n_items, n_segments, offsets.data(), offsets.data() + 1, 0, 32,
                (cudaStream_t)stream);
            CUDA_CHECK_LAST_ERROR(DeviceSegmentedRadixSort);
        },
        "keys_in"_a, "keys_out"_a, "offsets"_a, "cub_temp"_a, nb::kw_only(),
        "n_items"_a, "n_segments"_a, "stream"_a = 0);

    m.def(
        "csr_extract_dense",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> row_ids,
           gpu_array_f<double, Device> out, int n_target, int col_start,
           int col_stop, std::uintptr_t stream) {
            int tpb = round_up_to_warp(n_target);
            int blocks = (n_target + tpb - 1) / tpb;
            csr_extract_dense_kernel<<<blocks, tpb, 0, (cudaStream_t)stream>>>(
                data.data(), indices.data(), indptr.data(), row_ids.data(),
                out.data(), n_target, col_start, col_stop);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);
        },
        "data"_a, "indices"_a, "indptr"_a, "row_ids"_a, "out"_a, nb::kw_only(),
        "n_target"_a, "col_start"_a, "col_stop"_a, "stream"_a = 0);

    m.def(
        "csr_extract_dense_f32",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> row_ids,
           gpu_array_f<float, Device> out, int n_target, int col_start,
           int col_stop, std::uintptr_t stream) {
            int tpb = round_up_to_warp(n_target);
            int blocks = (n_target + tpb - 1) / tpb;
            csr_extract_dense_kernel<<<blocks, tpb, 0, (cudaStream_t)stream>>>(
                data.data(), indices.data(), indptr.data(), row_ids.data(),
                out.data(), n_target, col_start, col_stop);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);
        },
        "data"_a, "indices"_a, "indptr"_a, "row_ids"_a, "out"_a, nb::kw_only(),
        "n_target"_a, "col_start"_a, "col_stop"_a, "stream"_a = 0);

    // ---- Streaming pipelines ----

    m.def(
        "ovo_streaming_csr",
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

    m.def(
        "ovo_streaming_csc",
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
        "ovo_streaming",
        [](gpu_array_f<const float, Device> ref_sorted,
           gpu_array_f<const float, Device> grp_data,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,
           int n_cols, int n_groups, bool compute_tie_corr,
           int sub_batch_cols) {
            ovo_streaming_impl(ref_sorted.data(), grp_data.data(),
                               grp_offsets.data(), rank_sums.data(),
                               tie_corr.data(), n_ref, n_all_grp, n_cols,
                               n_groups, compute_tie_corr, sub_batch_cols);
        },
        "ref_sorted"_a, "grp_data"_a, "grp_offsets"_a, "rank_sums"_a,
        "tie_corr"_a, nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    // ---- Host-streaming pipelines (host inputs, device outputs) ----

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
           bool compute_nnz, int sub_batch_cols) {                             \
            ovo_streaming_csr_host_impl<InT, IndexT, IndptrT>(                 \
                h_data.data(), h_indices.data(), h_indptr.data(), n_full_rows, \
                h_ref_row_ids.data(), n_ref, h_grp_row_ids.data(),             \
                h_grp_offsets.data(), n_all_grp, n_test, d_rank_sums.data(),   \
                d_tie_corr.data(), d_group_sums.data(),                        \
                d_group_sq_sums.data(), d_group_nnz.data(), n_cols,            \
                n_groups_stats, compute_tie_corr, compute_sq_sums,             \
                compute_nnz, sub_batch_cols);                                  \
        },                                                                     \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_ref_row_ids"_a,            \
        "h_grp_row_ids"_a, "h_grp_offsets"_a, "d_rank_sums"_a, "d_tie_corr"_a, \
        "d_group_sums"_a, "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), \
        "n_full_rows"_a, "n_ref"_a, "n_all_grp"_a, "n_cols"_a, "n_test"_a,     \
        "n_groups_stats"_a, "compute_tie_corr"_a, "compute_sq_sums"_a = true,  \
        "compute_nnz"_a = true, "sub_batch_cols"_a = SUB_BATCH_COLS)

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

#define RSC_OVO_DENSE_HOST_BINDING(NAME, InT)                                 \
    m.def(                                                                    \
        NAME,                                                                 \
        [](host_array_2d<const InT> h_block,                                  \
           host_array<const int> h_ref_row_ids,                               \
           host_array<const int> h_grp_row_ids,                               \
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
            ovo_streaming_dense_host_impl<InT>(                               \
                h_block.data(), h_ref_row_ids.data(), h_grp_row_ids.data(),   \
                h_grp_offsets.data(), h_stats_codes.data(),                   \
                d_rank_sums.data(), d_tie_corr.data(), d_group_sums.data(),   \
                d_group_sq_sums.data(), d_group_nnz.data(), n_ref, n_all_grp, \
                n_rows, n_cols, n_groups, n_groups_stats, compute_tie_corr,   \
                compute_sq_sums, compute_nnz, sub_batch_cols);                \
        },                                                                    \
        "h_block"_a, "h_ref_row_ids"_a, "h_grp_row_ids"_a, "h_grp_offsets"_a, \
        "h_stats_codes"_a, "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a, \
        "d_group_sq_sums"_a, "d_group_nnz"_a, nb::kw_only(), "n_ref"_a,       \
        "n_all_grp"_a, "n_rows"_a, "n_cols"_a, "n_groups"_a,                  \
        "n_groups_stats"_a, "compute_tie_corr"_a, "compute_sq_sums"_a = true, \
        "compute_nnz"_a = true, "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVO_DENSE_HOST_BINDING("ovo_streaming_dense_host", float);
    RSC_OVO_DENSE_HOST_BINDING("ovo_streaming_dense_host_f64", double);
#undef RSC_OVO_DENSE_HOST_BINDING
}
