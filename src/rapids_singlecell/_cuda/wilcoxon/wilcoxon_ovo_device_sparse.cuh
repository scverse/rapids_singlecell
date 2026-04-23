#pragma once

/**
 * CSR-direct OVO streaming pipeline.
 *
 * One C++ call does everything: extract rows from CSR → sort → rank.
 * Per sub-batch of columns:
 *   1. Extract ref rows → dense f32 → CUB sort
 *   2. Extract grp rows → dense f32 → CUB sort (segmented by group)
 *   3. Binary search rank sums
 * Only ~(n_ref + n_all_grp) × sub_batch × 4B on GPU at a time.
 */
static void ovo_streaming_csr_impl(
    const float* csr_data, const int* csr_indices, const int* csr_indptr,
    const int* ref_row_ids, const int* grp_row_ids, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch: read group offsets to determine max group size ----
    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_tier1_config(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool use_tier1 = t1.any_above_t2 && t1.use_tier1;
    bool needs_tier3 = t1.any_above_t2 && !use_tier1;
    int padded_grp_size = t1.padded_grp_size;
    int tier1_tpb = t1.tier1_tpb;
    size_t tier1_smem = t1.tier1_smem;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (needs_tier3) {
        h_sort_group_ids = make_sort_group_ids(h_offsets.data(), n_groups,
                                               TIER2_GROUP_THRESHOLD);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_ref_items = (size_t)n_ref * sub_batch_cols;
    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;

    // CUB temp for ref sort (always needed) + grp sort (Tier 3 only)
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
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].ref_tie_sums = (t1.any_tier2 && compute_tie_corr)
                                   ? pool.alloc<double>(sub_batch_cols)
                                   : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (needs_tier3) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_seg = n_sort_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_seg);
            bufs[s].grp_seg_ends = pool.alloc<int>(max_seg);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
            bufs[s].grp_seg_ends = nullptr;
        }
    }

    int tpb_extract = round_up_to_warp(std::max(n_ref, n_all_grp));
    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_ref_items_actual = n_ref * sb_cols;
        int sb_grp_items_actual = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // ---- Extract + sort ref (always CUB) ----
        cudaMemsetAsync(buf.ref_dense, 0, sb_ref_items_actual * sizeof(float),
                        stream);
        {
            int blk = (n_ref + tpb_extract - 1) / tpb_extract;
            csr_extract_dense_kernel<<<blk, tpb_extract, 0, stream>>>(
                csr_data, csr_indices, csr_indptr, ref_row_ids, buf.ref_dense,
                n_ref, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);
        }
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortKeys(
                buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                sb_ref_items_actual, sb_cols, buf.ref_seg_offsets,
                buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // ---- Extract grp rows ----
        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_items_actual * sizeof(float),
                        stream);
        {
            int blk = (n_all_grp + tpb_extract - 1) / tpb_extract;
            csr_extract_dense_kernel<<<blk, tpb_extract, 0, stream>>>(
                csr_data, csr_indices, csr_indptr, grp_row_ids, buf.grp_dense,
                n_all_grp, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);
        }

        // Tier 0 handles groups ≤ TIER0_GROUP_THRESHOLD; Tier 1/3 handle
        // the rest.  Since each group owns its own rank_sums / tie_corr
        // row, the two kernels' writes interlace without conflict.
        int skip_le = 0;
        if (t1.use_tier0) {
            launch_tier0(buf.ref_sorted, buf.grp_dense, grp_offsets,
                         buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                         sb_cols, n_groups, compute_tie_corr, stream);
            if (t1.any_above_t0) skip_le = TIER0_GROUP_THRESHOLD;
        }
        if (t1.any_tier2) {
            if (compute_tie_corr) {
                launch_ref_tie_sums(buf.ref_sorted, buf.ref_tie_sums, n_ref,
                                    sb_cols, stream);
            }
            launch_tier2_medium(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.ref_tie_sums,
                buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                n_groups, compute_tie_corr, TIER0_GROUP_THRESHOLD, stream);
        }

        int upper_skip_le = t1.any_above_t2 ? TIER2_GROUP_THRESHOLD : skip_le;
        if (t1.any_above_t2 && use_tier1) {
            // ---- Tier 1: fused smem sort + binary search ----
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.sub_rank_sums,
                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size, upper_skip_le);
            CUDA_CHECK_LAST_ERROR(ovo_fused_sort_rank_kernel);
        } else if (needs_tier3) {
            // ---- Tier 3: CUB segmented sort + binary search ----
            int sb_grp_seg = n_sort_groups * sb_cols;
            {
                int blk = (sb_grp_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                build_tier3_seg_begin_end_offsets_kernel<<<blk, UTIL_BLOCK_SIZE,
                                                           0, stream>>>(
                    grp_offsets, d_sort_group_ids, buf.grp_seg_offsets,
                    buf.grp_seg_ends, n_all_grp, n_sort_groups, sb_cols);
                CUDA_CHECK_LAST_ERROR(build_tier3_seg_begin_end_offsets_kernel);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_items_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_ends, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, grp_offsets,
                    buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr, upper_skip_le);
                CUDA_CHECK_LAST_ERROR(batched_rank_sums_presorted_kernel);
            }
        }

        // ---- Scatter to global output ----
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

/**
 * CSC-direct OVO streaming pipeline.
 *
 * Like the CSR variant but extracts rows via a row-lookup map, avoiding
 * CSC→CSR conversion.  row_map_ref[row] = output index in ref block (-1 if
 * not a ref row); likewise for row_map_grp.
 */
static void ovo_streaming_csc_impl(
    const float* csc_data, const int* csc_indices, const int* csc_indptr,
    const int* ref_row_map, const int* grp_row_map, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch ----
    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_tier1_config(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool use_tier1 = t1.any_above_t2 && t1.use_tier1;
    bool needs_tier3 = t1.any_above_t2 && !use_tier1;
    int padded_grp_size = t1.padded_grp_size;
    int tier1_tpb = t1.tier1_tpb;
    size_t tier1_smem = t1.tier1_smem;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (needs_tier3) {
        h_sort_group_ids = make_sort_group_ids(h_offsets.data(), n_groups,
                                               TIER2_GROUP_THRESHOLD);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

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
    int* d_sort_group_ids = nullptr;
    if (needs_tier3) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                   h_sort_group_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }
    struct StreamBuf {
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].ref_tie_sums = (t1.any_tier2 && compute_tie_corr)
                                   ? pool.alloc<double>(sub_batch_cols)
                                   : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
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

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_ref_items_actual = n_ref * sb_cols;
        int sb_grp_items_actual = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // ---- Extract ref from CSC via row_map, then sort ----
        cudaMemsetAsync(buf.ref_dense, 0, sb_ref_items_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, ref_row_map, buf.ref_dense,
            n_ref, col);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortKeys(
                buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                sb_ref_items_actual, sb_cols, buf.ref_seg_offsets,
                buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // ---- Extract grp from CSC via row_map ----
        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_items_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, grp_row_map, buf.grp_dense,
            n_all_grp, col);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);

        int skip_le = 0;
        if (t1.use_tier0) {
            launch_tier0(buf.ref_sorted, buf.grp_dense, grp_offsets,
                         buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                         sb_cols, n_groups, compute_tie_corr, stream);
            if (t1.any_above_t0) skip_le = TIER0_GROUP_THRESHOLD;
        }
        if (t1.any_tier2) {
            if (compute_tie_corr) {
                launch_ref_tie_sums(buf.ref_sorted, buf.ref_tie_sums, n_ref,
                                    sb_cols, stream);
            }
            launch_tier2_medium(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.ref_tie_sums,
                buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                n_groups, compute_tie_corr, TIER0_GROUP_THRESHOLD, stream);
        }

        int upper_skip_le = t1.any_above_t2 ? TIER2_GROUP_THRESHOLD : skip_le;
        if (t1.any_above_t2 && use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.sub_rank_sums,
                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size, upper_skip_le);
            CUDA_CHECK_LAST_ERROR(ovo_fused_sort_rank_kernel);
        } else if (needs_tier3) {
            int sb_grp_seg = n_sort_groups * sb_cols;
            {
                int blk = (sb_grp_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                build_tier3_seg_begin_end_offsets_kernel<<<blk, UTIL_BLOCK_SIZE,
                                                           0, stream>>>(
                    grp_offsets, d_sort_group_ids, buf.grp_seg_offsets,
                    buf.grp_seg_ends, n_all_grp, n_sort_groups, sb_cols);
                CUDA_CHECK_LAST_ERROR(build_tier3_seg_begin_end_offsets_kernel);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_items_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_ends, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, grp_offsets,
                    buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr, upper_skip_le);
                CUDA_CHECK_LAST_ERROR(batched_rank_sums_presorted_kernel);
            }
        }

        // ---- Scatter to global output ----
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
