#pragma once

/**
 * Sparse-aware host-streaming CSC OVR pipeline.
 *
 * Like ovr_streaming_csc_host_impl but sorts only stored nonzeros per column
 * instead of extracting dense blocks.  GPU memory is O(max_batch_nnz) instead
 * of O(sub_batch * n_rows), and sort work is proportional to nnz, not n_rows.
 */
template <typename InT, typename IndptrT>
static void ovr_sparse_csc_host_streaming_impl(
    const InT* h_data, const int* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_sq_sums,
    double* d_group_nnz, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, bool compute_sq_sums, bool compute_nnz,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    // Find max nnz across any sub-batch
    size_t max_nnz = 0;
    for (int col = 0; col < n_cols; col += sub_batch_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        size_t nnz = (size_t)(h_indptr[col + sb_cols] - h_indptr[col]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

    // CUB temp size for max_nnz items
    size_t cub_temp_bytes = 0;
    if (max_nnz > 0) {
        auto* fk = reinterpret_cast<float*>(1);
        auto* iv = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, cub_temp_bytes, fk, fk, iv, iv, (int)max_nnz,
            sub_batch_cols, iv, iv + 1, BEGIN_BIT, END_BIT);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmPool pool;
    int* d_group_codes = pool.alloc<int>(n_rows);
    double* d_group_sizes = pool.alloc<double>(n_groups);
    struct StreamBuf {
        InT* d_sparse_data_orig;
        float* d_sparse_data_f32;
        int* d_sparse_indices;
        int* d_seg_offsets;
        float* keys_out;
        int* vals_out;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
        double* d_group_sums;
        double* d_group_sq_sums;
        double* d_group_nnz;
        double* d_nz_scratch;  // gmem-only; non-null when rank_use_gmem
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_sparse_data_orig = pool.alloc<InT>(max_nnz);
        bufs[s].d_sparse_data_f32 = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<int>(max_nnz);
        bufs[s].d_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].keys_out = pool.alloc<float>(max_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_nnz);
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

    // Transfer group codes + sizes once
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    // Pre-compute rebased per-batch offsets and upload once (avoids per-batch
    // H2D copy from a transient host buffer).
    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int sb = std::min(sub_batch_cols, n_cols - col_start);
        IndptrT ptr_start = h_indptr[col_start];
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        for (int i = 0; i <= sb; i++)
            off[i] = (int)(h_indptr[col_start + i] - ptr_start);
    }
    int* d_all_offsets =
        pool.alloc<int>((size_t)n_batches * (sub_batch_cols + 1));
    cudaMemcpy(d_all_offsets, h_all_offsets.data(),
               h_all_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int tpb = UTIL_BLOCK_SIZE;
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    size_t smem_cast = (size_t)n_groups * sizeof(double);
    if (compute_nnz) {
        smem_cast = (size_t)(3 * n_groups) * sizeof(double);
    } else if (compute_sq_sums) {
        smem_cast = (size_t)(2 * n_groups) * sizeof(double);
    }

    // In gmem mode the sparse rank kernel accumulates into rank_sums directly
    // and needs a per-stream nz_count scratch buffer sized (n_groups, sb_cols).
    for (int s = 0; s < n_streams; s++) {
        if (rank_use_gmem) {
            bufs[s].d_nz_scratch =
                pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        } else {
            bufs[s].d_nz_scratch = nullptr;
        }
    }

    // Pin only the host input arrays; outputs live on the device.
    size_t total_nnz = (size_t)h_indptr[n_cols];
    HostRegisterGuard _pin_data(const_cast<InT*>(h_data),
                                total_nnz * sizeof(InT));
    HostRegisterGuard _pin_indices(const_cast<int*>(h_indices),
                                   total_nnz * sizeof(int));

    cudaDeviceSynchronize();

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        IndptrT ptr_start = h_indptr[col];
        IndptrT ptr_end = h_indptr[col + sb_cols];
        int batch_nnz = (int)(ptr_end - ptr_start);

        // H2D: transfer sparse data for this column range (native dtype)
        if (batch_nnz > 0) {
            cudaMemcpyAsync(buf.d_sparse_data_orig, h_data + ptr_start,
                            (size_t)batch_nnz * sizeof(InT),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(buf.d_sparse_indices, h_indices + ptr_start,
                            (size_t)batch_nnz * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }

        // D2D: copy this batch's rebased offsets from the pre-uploaded buffer
        int* src = d_all_offsets + (size_t)batch_idx * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.d_seg_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Cast to float32 for sort + accumulate stats in float64
        ovr_cast_and_accumulate_sparse_kernel<InT>
            <<<sb_cols, tpb, smem_cast, stream>>>(
                buf.d_sparse_data_orig, buf.d_sparse_data_f32,
                buf.d_sparse_indices, buf.d_seg_offsets, d_group_codes,
                buf.d_group_sums, buf.d_group_sq_sums, buf.d_group_nnz, sb_cols,
                n_groups, compute_sq_sums, compute_nnz);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_sparse_kernel);

        // CUB sort only stored nonzeros (float32 keys)
        if (batch_nnz > 0) {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortPairs(
                buf.cub_temp, temp, buf.d_sparse_data_f32, buf.keys_out,
                buf.d_sparse_indices, buf.vals_out, batch_nnz, sb_cols,
                buf.d_seg_offsets, buf.d_seg_offsets + 1, BEGIN_BIT, END_BIT,
                stream);
        }

        // Sparse rank kernel (stats already captured above)
        if (rank_use_gmem) {
            cudaMemsetAsync(buf.d_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
            cudaMemsetAsync(buf.d_nz_scratch, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_sparse_ovr_kernel<<<sb_cols, tpb, smem_bytes, stream>>>(
            buf.keys_out, buf.vals_out, buf.d_seg_offsets, d_group_codes,
            d_group_sizes, buf.d_rank_sums, buf.d_tie_corr, buf.d_nz_scratch,
            n_rows, sb_cols, n_groups, compute_tie_corr, rank_use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_sparse_ovr_kernel);

        // D2D: scatter sub-batch results into caller's GPU buffers
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
                std::string("CUDA error in sparse host CSC streaming: ") +
                cudaGetErrorString(err));
    }

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

// ============================================================================
// Sparse-aware CSC OVR streaming (sort only stored nonzeros)
// ============================================================================

static void ovr_sparse_csc_streaming_impl(
    const float* csc_data, const int* csc_indices, const int* csc_indptr,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Read indptr to host for batch planning
    std::vector<int> h_indptr(n_cols + 1);
    cudaMemcpy(h_indptr.data(), csc_indptr, (n_cols + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    // Find max nnz across any sub-batch for buffer sizing
    size_t max_nnz = 0;
    for (int col = 0; col < n_cols; col += sub_batch_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        size_t nnz = (size_t)(h_indptr[col + sb_cols] - h_indptr[col]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

    // CUB temp size for max_nnz items
    size_t cub_temp_bytes = 0;
    if (max_nnz > 0) {
        auto* fk = reinterpret_cast<float*>(1);
        auto* iv = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, cub_temp_bytes, fk, fk, iv, iv, (int)max_nnz,
            sub_batch_cols, iv, iv + 1, BEGIN_BIT, END_BIT);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    int tpb = UTIL_BLOCK_SIZE;
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);

    RmmPool pool;
    struct StreamBuf {
        float* keys_out;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* d_nz_scratch;  // gmem-only
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].keys_out = pool.alloc<float>(max_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_nnz);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].d_nz_scratch =
            rank_use_gmem
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
    }

    cudaDeviceSynchronize();

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        int ptr_start = h_indptr[col];
        int ptr_end = h_indptr[col + sb_cols];
        int batch_nnz = ptr_end - ptr_start;

        // Compute rebased segment offsets on GPU (avoids host pinned-buffer
        // race)
        {
            int count = sb_cols + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                csc_indptr, buf.seg_offsets, col, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        // Sort only stored values (keys=data, vals=row_indices)
        if (batch_nnz > 0) {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortPairs(
                buf.cub_temp, temp, csc_data + ptr_start, buf.keys_out,
                csc_indices + ptr_start, buf.vals_out, batch_nnz, sb_cols,
                buf.seg_offsets, buf.seg_offsets + 1, BEGIN_BIT, END_BIT,
                stream);
        }

        // Sparse rank kernel (handles implicit zeros analytically)
        if (rank_use_gmem) {
            cudaMemsetAsync(buf.sub_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
            cudaMemsetAsync(buf.d_nz_scratch, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_sparse_ovr_kernel<<<sb_cols, tpb, smem_bytes, stream>>>(
            buf.keys_out, buf.vals_out, buf.seg_offsets, group_codes,
            group_sizes, buf.sub_rank_sums, buf.sub_tie_corr, buf.d_nz_scratch,
            n_rows, sb_cols, n_groups, compute_tie_corr, rank_use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_sparse_ovr_kernel);

        // Scatter results to global output
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

    for (int s = 0; s < n_streams; s++) {
        cudaError_t err = cudaStreamSynchronize(streams[s]);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDA error in sparse ovr streaming: ") +
                cudaGetErrorString(err));
    }

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

// ============================================================================
// Sparse-aware CSR OVR streaming (partial CSR→CSC transpose per sub-batch)
// ============================================================================

/**
 * Sparse-aware OVR streaming pipeline for GPU CSR data.
 *
 * Phase 0: One histogram kernel counts nnz per column. D2H + host prefix sums
 *          give exact per-batch nnz and max_batch_nnz for buffer sizing.
 * Phase 1: Allocate per-stream buffers sized to max_batch_nnz.
 * Phase 2: For each sub-batch: scatter CSR→CSC (partial transpose via
 *          atomics) → CUB sort only nonzeros → sparse rank kernel.
 *
 * Compared to the dense CSR path, sort work drops by ~1/sparsity.
 */
static void ovr_sparse_csr_streaming_impl(
    const float* csr_data, const int* csr_indices, const int* csr_indptr,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // ---- Phase 0: Planning — count nnz per column via histogram ----
    RmmPool pool;
    int* d_col_counts = pool.alloc<int>(n_cols);
    cudaMemset(d_col_counts, 0, n_cols * sizeof(int));
    {
        int blocks = (n_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
        csr_col_histogram_kernel<<<blocks, UTIL_BLOCK_SIZE>>>(
            csr_indices, csr_indptr, d_col_counts, n_rows, n_cols);
        CUDA_CHECK_LAST_ERROR(csr_col_histogram_kernel);
    }
    std::vector<int> h_col_counts(n_cols);
    cudaMemcpy(h_col_counts.data(), d_col_counts, n_cols * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Per-batch prefix sums on host
    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    size_t max_batch_nnz = 0;

    // Flat array: n_batches × (sub_batch_cols + 1) offsets
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    std::vector<size_t> h_batch_nnz(n_batches);

    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int sb_cols = std::min(sub_batch_cols, n_cols - col_start);
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        off[0] = 0;
        for (int i = 0; i < sb_cols; i++)
            off[i + 1] = off[i] + h_col_counts[col_start + i];
        h_batch_nnz[b] = (size_t)off[sb_cols];
        if (h_batch_nnz[b] > max_batch_nnz) max_batch_nnz = h_batch_nnz[b];
    }

    // Upload all batch offsets to GPU in one shot (~20 KB)
    int* d_all_offsets =
        pool.alloc<int>((size_t)n_batches * (sub_batch_cols + 1));
    cudaMemcpy(d_all_offsets, h_all_offsets.data(),
               h_all_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // ---- Phase 1: Allocate per-stream buffers ----
    size_t cub_temp_bytes = 0;
    if (max_batch_nnz > 0) {
        auto* fk = reinterpret_cast<float*>(1);
        auto* iv = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, cub_temp_bytes, fk, fk, iv, iv, (int)max_batch_nnz,
            sub_batch_cols, iv, iv + 1, BEGIN_BIT, END_BIT);
    }

    int n_streams = N_STREAMS;
    if (n_batches < n_streams) n_streams = n_batches;

    // CSR path needs 4 sort arrays per stream (scatter intermediates +
    // CUB output).  Fit stream count to available GPU memory.
    size_t per_stream_bytes =
        max_batch_nnz * (2 * sizeof(float) + 2 * sizeof(int)) +
        (sub_batch_cols + 1 + sub_batch_cols) * sizeof(int) + cub_temp_bytes +
        (size_t)n_groups * sub_batch_cols * sizeof(double) +
        sub_batch_cols * sizeof(double);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    constexpr double MEM_BUDGET_FRAC = 0.8;
    size_t budget = (size_t)(free_mem * MEM_BUDGET_FRAC);
    while (n_streams > 1 && (size_t)n_streams * per_stream_bytes > budget)
        n_streams--;

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    int tpb = UTIL_BLOCK_SIZE;
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    int scatter_blocks = (n_rows + tpb - 1) / tpb;

    struct StreamBuf {
        int* col_offsets;  // [sub_batch_cols + 1]  CSC-style offsets
        int* write_pos;    // [sub_batch_cols]      atomic write counters
        float* csc_vals;   // [max_batch_nnz]       transposed values
        int* csc_row_idx;  // [max_batch_nnz]       transposed row indices
        float* keys_out;   // [max_batch_nnz]       CUB sort output
        int* vals_out;     // [max_batch_nnz]       CUB sort output
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* d_nz_scratch;  // gmem-only
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].col_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].write_pos = pool.alloc<int>(sub_batch_cols);
        bufs[s].csc_vals = pool.alloc<float>(max_batch_nnz);
        bufs[s].csc_row_idx = pool.alloc<int>(max_batch_nnz);
        bufs[s].keys_out = pool.alloc<float>(max_batch_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_batch_nnz);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].d_nz_scratch =
            rank_use_gmem
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
    }

    cudaDeviceSynchronize();

    // ---- Phase 2: Stream loop ----
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = b % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];
        int batch_nnz = (int)h_batch_nnz[b];

        // D2D copy pre-computed col_offsets for this batch
        int* src = d_all_offsets + (size_t)b * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.col_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Initialize write_pos = col_offsets[0..sb_cols-1] (same D2D source)
        cudaMemcpyAsync(buf.write_pos, src, sb_cols * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        if (batch_nnz > 0) {
            // Scatter CSR → CSC layout for this sub-batch
            csr_scatter_to_csc_kernel<<<scatter_blocks, tpb, 0, stream>>>(
                csr_data, csr_indices, csr_indptr, buf.write_pos, buf.csc_vals,
                buf.csc_row_idx, n_rows, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);

            // CUB sort only the nonzeros
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortPairs(
                buf.cub_temp, temp, buf.csc_vals, buf.keys_out, buf.csc_row_idx,
                buf.vals_out, batch_nnz, sb_cols, buf.col_offsets,
                buf.col_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // Sparse rank kernel (handles implicit zeros analytically)
        if (rank_use_gmem) {
            cudaMemsetAsync(buf.sub_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
            cudaMemsetAsync(buf.d_nz_scratch, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_sparse_ovr_kernel<<<sb_cols, tpb, smem_bytes, stream>>>(
            buf.keys_out, buf.vals_out, buf.col_offsets, group_codes,
            group_sizes, buf.sub_rank_sums, buf.sub_tie_corr, buf.d_nz_scratch,
            n_rows, sb_cols, n_groups, compute_tie_corr, rank_use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_sparse_ovr_kernel);

        // Scatter results to global output
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
    }

    for (int s = 0; s < n_streams; s++) {
        cudaError_t err = cudaStreamSynchronize(streams[s]);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDA error in sparse CSR ovr streaming: ") +
                cudaGetErrorString(err));
    }

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}
