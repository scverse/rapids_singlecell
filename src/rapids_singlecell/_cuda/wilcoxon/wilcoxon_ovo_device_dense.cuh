#pragma once

/**
 * Streaming OVO pipeline.
 *
 * Takes pre-sorted reference (float32 F-order), unsorted group data (float32
 * F-order with group offsets), and produces rank_sums + tie_corr.
 *
 * For each sub-batch of columns:
 *   1. CUB segmented sort-keys of group data (one segment per group per col)
 *   2. batched_rank_sums_presorted_kernel (binary search in sorted ref)
 */
static void ovo_streaming_impl(const float* ref_sorted, const float* grp_data,
                               const int* grp_offsets, double* rank_sums,
                               double* tie_corr, int n_ref, int n_all_grp,
                               int n_cols, int n_groups, bool compute_tie_corr,
                               int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch ----
    std::vector<int> h_off(n_groups + 1);
    cudaMemcpy(h_off.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_tier1_config(h_off.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool use_tier1 = t1.any_above_t2 && t1.use_tier1;
    bool needs_tier3 = t1.any_above_t2 && !use_tier1;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (needs_tier3) {
        h_sort_group_ids =
            make_sort_group_ids(h_off.data(), n_groups, TIER2_GROUP_THRESHOLD);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    size_t cub_temp_bytes = 0;
    if (needs_tier3) {
        int max_n_seg = n_sort_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_temp_bytes, fk, fk, (int)sub_grp_items, max_n_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    // Allocate per-stream buffers via RMM pool
    RmmPool pool;
    int* d_sort_group_ids = nullptr;
    if (needs_tier3) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                   h_sort_group_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }
    struct StreamBuf {
        float* grp_sorted;
        int* seg_offsets;
        int* seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        if (needs_tier3) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_n_seg = n_sort_groups * sub_batch_cols;
            bufs[s].seg_offsets = pool.alloc<int>(max_n_seg);
            bufs[s].seg_ends = pool.alloc<int>(max_n_seg);
            bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].seg_offsets = nullptr;
            bufs[s].seg_ends = nullptr;
            bufs[s].cub_temp = nullptr;
        }
        bufs[s].ref_tie_sums = (t1.any_tier2 && compute_tie_corr)
                                   ? pool.alloc<double>(sub_batch_cols)
                                   : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_grp_items = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        const float* grp_in = grp_data + (long long)col * n_all_grp;
        const float* ref_sub = ref_sorted + (long long)col * n_ref;

        int skip_le = 0;
        if (t1.use_tier0) {
            launch_tier0(ref_sub, grp_in, grp_offsets, buf.sub_rank_sums,
                         buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                         compute_tie_corr, stream);
            if (t1.any_above_t0) skip_le = TIER0_GROUP_THRESHOLD;
        }
        if (t1.any_tier2) {
            if (compute_tie_corr) {
                launch_ref_tie_sums(ref_sub, buf.ref_tie_sums, n_ref, sb_cols,
                                    stream);
            }
            launch_tier2_medium(ref_sub, grp_in, grp_offsets, buf.ref_tie_sums,
                                buf.sub_rank_sums, buf.sub_tie_corr, n_ref,
                                n_all_grp, sb_cols, n_groups, compute_tie_corr,
                                TIER0_GROUP_THRESHOLD, stream);
        }

        int upper_skip_le = t1.any_above_t2 ? TIER2_GROUP_THRESHOLD : skip_le;
        if (t1.any_above_t2 && use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, t1.tier1_tpb, t1.tier1_smem,
                                         stream>>>(
                ref_sub, grp_in, grp_offsets, buf.sub_rank_sums,
                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, t1.padded_grp_size, upper_skip_le);
            CUDA_CHECK_LAST_ERROR(ovo_fused_sort_rank_kernel);
        } else if (needs_tier3) {
            int sb_n_seg = n_sort_groups * sb_cols;
            {
                int blk = (sb_n_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                build_tier3_seg_begin_end_offsets_kernel<<<blk, UTIL_BLOCK_SIZE,
                                                           0, stream>>>(
                    grp_offsets, d_sort_group_ids, buf.seg_offsets,
                    buf.seg_ends, n_all_grp, n_sort_groups, sb_cols);
                CUDA_CHECK_LAST_ERROR(build_tier3_seg_begin_end_offsets_kernel);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, grp_in, buf.grp_sorted, sb_grp_items,
                    sb_n_seg, buf.seg_offsets, buf.seg_ends, BEGIN_BIT, END_BIT,
                    stream);
                CUDA_CHECK_LAST_ERROR(DeviceSegmentedRadixSort);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    ref_sub, buf.grp_sorted, grp_offsets, buf.sub_rank_sums,
                    buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                    compute_tie_corr, upper_skip_le);
                CUDA_CHECK_LAST_ERROR(batched_rank_sums_presorted_kernel);
            }
        }

        // Scatter sub-batch results to global output
        cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                          buf.sub_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpy2DAsync(tie_corr + col, n_cols * sizeof(double),
                              buf.sub_tie_corr, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
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
