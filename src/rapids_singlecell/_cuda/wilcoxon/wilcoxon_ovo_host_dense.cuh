#pragma once

/**
 * Gather specific rows from a dense F-order block into a smaller dense block.
 * Grid: (n_cols,), Block: 256.
 * row_ids[i] = original row index → output row i.
 */
__global__ void dense_gather_rows_kernel(const float* __restrict__ in,
                                         const int* __restrict__ row_ids,
                                         float* __restrict__ out, int n_rows_in,
                                         int n_target, int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    const float* in_col = in + (long long)col * n_rows_in;
    float* out_col = out + (long long)col * n_target;
    for (int i = threadIdx.x; i < n_target; i += blockDim.x) {
        out_col[i] = in_col[row_ids[i]];
    }
}

/**
 * Host-streaming dense OVO pipeline.
 *
 * Dense F-order float32 lives on host.  Sub-batches of columns are H2D
 * transferred, then ref/grp rows are gathered, sorted, and ranked.
 * Results D2H per sub-batch.
 */
template <typename InT>
static void ovo_streaming_dense_host_impl(
    const InT* h_block, const int* h_ref_row_ids, const int* h_grp_row_ids,
    const int* h_grp_offsets, const int* h_stats_codes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_sq_sums,
    double* d_group_nnz, int n_ref, int n_all_grp, int n_rows, int n_cols,
    int n_groups, int n_groups_stats, bool compute_tie_corr,
    bool compute_sq_sums, bool compute_nnz, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch from host offsets ----
    auto t1 = make_tier1_config(h_grp_offsets, n_groups);
    int max_grp_size = t1.max_grp_size;
    bool use_tier1 = t1.any_above_t2 && t1.use_tier1;
    bool needs_tier3 = t1.any_above_t2 && !use_tier1;
    int padded_grp_size = t1.padded_grp_size;
    int tier1_tpb = t1.tier1_tpb;
    size_t tier1_smem = t1.tier1_smem;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (needs_tier3) {
        h_sort_group_ids =
            make_sort_group_ids(h_grp_offsets, n_groups, TIER2_GROUP_THRESHOLD);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_dense = (size_t)n_rows * sub_batch_cols;
    size_t sub_ref_items = (size_t)n_ref * sub_batch_cols;
    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;

    // CUB temp
    size_t cub_ref_bytes = 0;
    {
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_ref_bytes, fk, fk, (int)sub_ref_items, sub_batch_cols,
            doff, doff + 1, BEGIN_BIT, END_BIT);
    }
    size_t cub_temp_bytes = cub_ref_bytes;
    if (needs_tier3) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg = n_sort_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_grp_bytes, fk, fk, (int)sub_grp_items, max_grp_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmPool pool;

    // GPU copies of row_ids + group offsets + stats codes (uploaded once)
    int* d_ref_row_ids = pool.alloc<int>(n_ref);
    int* d_grp_row_ids = pool.alloc<int>(n_all_grp);
    int* d_grp_offsets = pool.alloc<int>(n_groups + 1);
    int* d_stats_codes = pool.alloc<int>(n_rows);
    int* d_sort_group_ids = nullptr;
    cudaMemcpy(d_ref_row_ids, h_ref_row_ids, n_ref * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_row_ids, h_grp_row_ids, n_all_grp * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets, h_grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stats_codes, h_stats_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    if (needs_tier3) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                   h_sort_group_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    struct StreamBuf {
        InT* d_block_orig;
        float* d_block_f32;
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* d_rank_sums;
        double* d_tie_corr;
        double* d_group_sums;
        double* d_group_sq_sums;
        double* d_group_nnz;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_block_orig = pool.alloc<InT>(sub_dense);
        bufs[s].d_block_f32 = pool.alloc<float>(sub_dense);
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].ref_tie_sums = (t1.any_tier2 && compute_tie_corr)
                                   ? pool.alloc<double>(sub_batch_cols)
                                   : nullptr;
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_group_sums =
            pool.alloc<double>((size_t)n_groups_stats * sub_batch_cols);
        bufs[s].d_group_sq_sums = pool.alloc<double>(
            compute_sq_sums ? (size_t)n_groups_stats * sub_batch_cols : 1);
        bufs[s].d_group_nnz = pool.alloc<double>(
            compute_nnz ? (size_t)n_groups_stats * sub_batch_cols : 1);
        if (needs_tier3) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = n_sort_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_grp_seg);
            bufs[s].grp_seg_ends = pool.alloc<int>(max_grp_seg);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
            bufs[s].grp_seg_ends = nullptr;
        }
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));
    size_t smem_cast = (size_t)(3 * n_groups_stats) * sizeof(double);

    // Pin only the host input; outputs live on the device.
    HostRegisterGuard _pin_block(const_cast<InT*>(h_block),
                                 (size_t)n_rows * n_cols * sizeof(InT));

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_dense = n_rows * sb_cols;
        int sb_ref_actual = n_ref * sb_cols;
        int sb_grp_actual = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // ---- H2D: dense column sub-batch (F-order, native dtype) ----
        cudaMemcpyAsync(buf.d_block_orig, h_block + (long long)col * n_rows,
                        sb_dense * sizeof(InT), cudaMemcpyHostToDevice, stream);

        // ---- Cast to float32 for sort + accumulate stats in float64 ----
        ovr_cast_and_accumulate_dense_kernel<InT>
            <<<sb_cols, UTIL_BLOCK_SIZE, smem_cast, stream>>>(
                buf.d_block_orig, buf.d_block_f32, d_stats_codes,
                buf.d_group_sums, buf.d_group_sq_sums, buf.d_group_nnz, n_rows,
                sb_cols, n_groups_stats, compute_sq_sums, compute_nnz);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_dense_kernel);

        // ---- Gather ref rows, sort ----
        dense_gather_rows_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.d_block_f32, d_ref_row_ids, buf.ref_dense, n_rows, n_ref,
            sb_cols);
        CUDA_CHECK_LAST_ERROR(dense_gather_rows_kernel);
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortKeys(
                buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                sb_ref_actual, sb_cols, buf.ref_seg_offsets,
                buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // ---- Gather grp rows ----
        dense_gather_rows_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.d_block_f32, d_grp_row_ids, buf.grp_dense, n_rows, n_all_grp,
            sb_cols);
        CUDA_CHECK_LAST_ERROR(dense_gather_rows_kernel);

        // ---- Tier dispatch: sort grp + rank ----
        int skip_le = 0;
        if (t1.use_tier0) {
            launch_tier0(buf.ref_sorted, buf.grp_dense, d_grp_offsets,
                         buf.d_rank_sums, buf.d_tie_corr, n_ref, n_all_grp,
                         sb_cols, n_groups, compute_tie_corr, stream);
            if (t1.any_above_t0) skip_le = TIER0_GROUP_THRESHOLD;
        }
        if (t1.any_tier2) {
            if (compute_tie_corr) {
                launch_ref_tie_sums(buf.ref_sorted, buf.ref_tie_sums, n_ref,
                                    sb_cols, stream);
            }
            launch_tier2_medium(
                buf.ref_sorted, buf.grp_dense, d_grp_offsets, buf.ref_tie_sums,
                buf.d_rank_sums, buf.d_tie_corr, n_ref, n_all_grp, sb_cols,
                n_groups, compute_tie_corr, TIER0_GROUP_THRESHOLD, stream);
        }

        int upper_skip_le = t1.any_above_t2 ? TIER2_GROUP_THRESHOLD : skip_le;
        if (t1.any_above_t2 && use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, d_grp_offsets, buf.d_rank_sums,
                buf.d_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size, upper_skip_le);
            CUDA_CHECK_LAST_ERROR(ovo_fused_sort_rank_kernel);
        } else if (needs_tier3) {
            int sb_grp_seg = n_sort_groups * sb_cols;
            {
                int blk = (sb_grp_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                build_tier3_seg_begin_end_offsets_kernel<<<blk, UTIL_BLOCK_SIZE,
                                                           0, stream>>>(
                    d_grp_offsets, d_sort_group_ids, buf.grp_seg_offsets,
                    buf.grp_seg_ends, n_all_grp, n_sort_groups, sb_cols);
                CUDA_CHECK_LAST_ERROR(build_tier3_seg_begin_end_offsets_kernel);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_ends, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, d_grp_offsets,
                    buf.d_rank_sums, buf.d_tie_corr, n_ref, n_all_grp, sb_cols,
                    n_groups, compute_tie_corr, upper_skip_le);
                CUDA_CHECK_LAST_ERROR(batched_rank_sums_presorted_kernel);
            }
        }

        // ---- D2D: scatter sub-batch results into caller's GPU buffers ----
        cudaMemcpy2DAsync(d_rank_sums + col, n_cols * sizeof(double),
                          buf.d_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpy2DAsync(d_tie_corr + col, n_cols * sizeof(double),
                              buf.d_tie_corr, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream);
        }
        cudaMemcpy2DAsync(d_group_sums + col, n_cols * sizeof(double),
                          buf.d_group_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups_stats,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_sq_sums) {
            cudaMemcpy2DAsync(d_group_sq_sums + col, n_cols * sizeof(double),
                              buf.d_group_sq_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups_stats,
                              cudaMemcpyDeviceToDevice, stream);
        }
        if (compute_nnz) {
            cudaMemcpy2DAsync(d_group_nnz + col, n_cols * sizeof(double),
                              buf.d_group_nnz, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups_stats,
                              cudaMemcpyDeviceToDevice, stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    for (int s = 0; s < n_streams; s++) {
        cudaError_t err = cudaStreamSynchronize(streams[s]);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDA error in wilcoxon streaming: ") +
                cudaGetErrorString(err));
    }

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

// ============================================================================
// Nanobind module
// ============================================================================
