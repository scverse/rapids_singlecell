#pragma once

/**
 * Streaming OVR pipeline.
 *
 * Takes a dense F-order float32 block (n_rows, n_cols) + int32 group_codes,
 * splits columns into sub-batches across multiple CUDA streams, and for each:
 *   1. CUB SortPairs (float32 keys + int32 row indices)
 *   2. Fused rank_sums_from_sorted_kernel
 *
 * Output: rank_sums (n_groups, n_cols) + tie_corr (n_cols), both float64.
 */
static void ovr_streaming_impl(const float* block, const int* group_codes,
                               double* rank_sums, double* tie_corr, int n_rows,
                               int n_cols, int n_groups, bool compute_tie_corr,
                               int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_items = (size_t)n_rows * sub_batch_cols;
    size_t cub_temp_bytes = 0;
    {
        auto* fk = reinterpret_cast<float*>(1);
        auto* iv = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, cub_temp_bytes, fk, fk, iv, iv, (int)sub_items,
            sub_batch_cols, iv, iv + 1, BEGIN_BIT, END_BIT);
    }

    // Create streams
    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    // Allocate per-stream buffers via RMM pool
    RmmPool pool;
    struct StreamBuf {
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
    }

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);

    // Process sub-batches round-robin across streams
    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // Fill segment offsets + row indices
        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.vals_in, n_rows, sb_cols);
        CUDA_CHECK_LAST_ERROR(fill_row_indices_kernel);

        // Sort: keys = block columns [col, col+sb_cols), already F-order
        const float* keys_in = block + (long long)col * n_rows;
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, keys_in, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums into sub-batch buffer
        if (use_gmem) {
            cudaMemsetAsync(buf.sub_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, buf.sub_rank_sums,
            buf.sub_tie_corr, n_rows, sb_cols, n_groups, compute_tie_corr,
            use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_from_sorted_kernel);

        // Copy sub-batch results to global output (row-major scatter)
        // rank_sums is (n_groups, n_cols) row-major: group g, col c →
        // [g*n_cols+c] sub output is (n_groups, sb_cols): group g, local col lc
        // → [g*sb_cols+lc]
        cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                          buf.sub_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + col, buf.sub_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    // Sync all streams
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
 * Host-streaming dense OVR pipeline.
 *
 * Templated on the host dtype (InT = float or double).  Each sub-batch is
 * copied to the device in its native dtype once; a fused cast+accumulate
 * kernel writes a float32 view for the sort and accumulates per-group
 * sum/sum-sq/nnz in float64 from the original-precision values.  The
 * existing sort + rank pipeline then runs on the float32 keys.
 *
 * Output pointers ({d_rank_sums, d_tie_corr, d_group_sums, d_group_sq_sums,
 * d_group_nnz}) point to caller-provided CuPy memory of the full output
 * shape; sub-batch kernels scatter directly into them via D2D.
 *
 * GPU memory stays at O(sub_batch * n_rows), now with a small extra
 * InT-sized sub-batch buffer per stream.
 */
template <typename InT>
static void ovr_streaming_dense_host_impl(
    const InT* h_block, const int* h_group_codes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_sq_sums,
    double* d_group_nnz, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, bool compute_sq_sums, bool compute_nnz,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_items = (size_t)n_rows * sub_batch_cols;
    size_t cub_temp_bytes = 0;
    {
        auto* fk = reinterpret_cast<float*>(1);
        auto* iv = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, cub_temp_bytes, fk, fk, iv, iv, (int)sub_items,
            sub_batch_cols, iv, iv + 1, BEGIN_BIT, END_BIT);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    // Allocate per-stream buffers via RMM pool
    RmmPool pool;
    int* d_group_codes = pool.alloc<int>(n_rows);
    struct StreamBuf {
        InT* d_block_orig;
        float* d_block_f32;
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
        double* d_group_sums;
        double* d_group_sq_sums;
        double* d_group_nnz;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_block_orig = pool.alloc<InT>(sub_items);
        bufs[s].d_block_f32 = pool.alloc<float>(sub_items);
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].d_group_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_group_sq_sums =
            compute_sq_sums
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
        bufs[s].d_group_nnz =
            compute_nnz ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                        : nullptr;
    }

    // Group codes on GPU (transferred once)
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);
    int tpb_cast = UTIL_BLOCK_SIZE;
    size_t smem_cast = (size_t)n_groups * sizeof(double);
    if (compute_nnz) {
        smem_cast = (size_t)(3 * n_groups) * sizeof(double);
    } else if (compute_sq_sums) {
        smem_cast = (size_t)(2 * n_groups) * sizeof(double);
    }

    // Pin only the host input.  Outputs live on the device (caller-owned).
    HostRegisterGuard _pin_block(const_cast<InT*>(h_block),
                                 (size_t)n_rows * n_cols * sizeof(InT));

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // H2D: column sub-batch in native dtype (F-order → contiguous)
        cudaMemcpyAsync(buf.d_block_orig, h_block + (long long)col * n_rows,
                        sb_items * sizeof(InT), cudaMemcpyHostToDevice, stream);

        // Cast to float32 for sort + accumulate stats in float64
        ovr_cast_and_accumulate_dense_kernel<InT>
            <<<sb_cols, tpb_cast, smem_cast, stream>>>(
                buf.d_block_orig, buf.d_block_f32, d_group_codes,
                buf.d_group_sums, buf.d_group_sq_sums, buf.d_group_nnz, n_rows,
                sb_cols, n_groups, compute_sq_sums, compute_nnz);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_dense_kernel);

        // Fill segment offsets + row indices
        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.vals_in, n_rows, sb_cols);
        CUDA_CHECK_LAST_ERROR(fill_row_indices_kernel);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.d_block_f32, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums (stats already captured by the cast kernel)
        if (use_gmem) {
            cudaMemsetAsync(buf.d_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, d_group_codes, buf.d_rank_sums,
            buf.d_tie_corr, n_rows, sb_cols, n_groups, compute_tie_corr,
            use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_from_sorted_kernel);

        // D2D: scatter sub-batch results into the caller's GPU buffers
        cudaMemcpy2DAsync(d_rank_sums + col, n_cols * sizeof(double),
                          buf.d_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(d_tie_corr + col, buf.d_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }
        cudaMemcpy2DAsync(d_group_sums + col, n_cols * sizeof(double),
                          buf.d_group_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_sq_sums) {
            cudaMemcpy2DAsync(d_group_sq_sums + col, n_cols * sizeof(double),
                              buf.d_group_sq_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream);
        }
        if (compute_nnz) {
            cudaMemcpy2DAsync(d_group_nnz + col, n_cols * sizeof(double),
                              buf.d_group_nnz, sb_cols * sizeof(double),
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
