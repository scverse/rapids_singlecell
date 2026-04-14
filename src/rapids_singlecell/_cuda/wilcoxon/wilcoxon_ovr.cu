#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_common.cuh"
#include "kernels_wilcoxon.cuh"

using namespace nb::literals;

/**
 * Decide whether to use shared or global memory for OVR rank accumulators.
 * Returns the smem size to request and sets use_gmem accordingly.
 */
static size_t ovr_smem_config(int n_groups, bool& use_gmem) {
    size_t need = (size_t)(4 * n_groups + 32) * sizeof(double);
    static int max_smem = -1;
    if (max_smem < 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock,
                               device);
    }
    if ((int)need <= max_smem) {
        use_gmem = false;
        return need;
    }
    // Fall back to global memory accumulators; only need warp buf in smem
    use_gmem = true;
    return 32 * sizeof(double);
}

/**
 * Extract dense F-order float32 block from CSC.
 * Column range [col_start, col_stop).
 * One block per column, threads scatter nonzeros.
 * Output must be pre-zeroed.
 */
__global__ void csc_extract_f32_kernel(const float* __restrict__ data,
                                       const int* __restrict__ indices,
                                       const int* __restrict__ indptr,
                                       float* __restrict__ out, int n_rows,
                                       int col_start) {
    int col_local = blockIdx.x;
    int col = col_start + col_local;

    int start = indptr[col];
    int end = indptr[col + 1];

    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        int row = indices[p];
        out[(long long)col_local * n_rows + row] = data[p];
    }
}

/**
 * Fill sort values with row indices [0,1,...,n_rows-1] per column.
 * Grid: (n_cols,), block: 256 threads.
 */
__global__ void fill_row_indices_kernel(int* __restrict__ vals, int n_rows,
                                        int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    int* out = vals + (long long)col * n_rows;
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        out[i] = i;
    }
}

/**
 * Launch csr_extract_dense_kernel for ALL rows of a CSR matrix.
 * Creates a temporary identity row_ids array [0,1,...,n_rows-1].
 */
static void csr_extract_all_rows(const float* data, const int* indices,
                                 const int* indptr, float* out, int n_rows,
                                 int col_start, int col_stop, RmmPool& pool,
                                 cudaStream_t stream) {
    int* row_ids = pool.alloc<int>(n_rows);
    fill_row_indices_kernel<<<1, 256, 0, stream>>>(row_ids, n_rows, 1);
    int tpb = round_up_to_warp(n_rows);
    int blk = (n_rows + tpb - 1) / tpb;
    csr_extract_dense_kernel<<<blk, tpb, 0, stream>>>(
        data, indices, indptr, row_ids, out, n_rows, col_start, col_stop);
}

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
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

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
            buf.sub_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false, use_gmem);

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
 * CSR-direct OVR streaming pipeline.
 *
 * Takes GPU CSR arrays directly — no CSR→CSC conversion needed.
 * For each sub-batch: extract dense columns from CSR → sort → rank.
 * Everything on one GPU with multi-stream overlap.
 */
static void ovr_streaming_csr_impl(
    const float* csr_data, const int* csr_indices, const int* csr_indptr,
    const int* group_codes, double* rank_sums, double* tie_corr, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, int sub_batch_cols) {
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
    struct StreamBuf {
        float* dense;
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
        bufs[s].dense = pool.alloc<float>(sub_items);
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

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // Zero dense buffer
        cudaMemsetAsync(buf.dense, 0, sb_items * sizeof(float), stream);

        // Extract dense columns from CSR (all rows)
        csr_extract_all_rows(csr_data, csr_indices, csr_indptr, buf.dense,
                             n_rows, col, col + sb_cols, pool, stream);

        // Fill segment offsets + row indices
        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.dense, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        if (use_gmem) {
            cudaMemsetAsync(buf.sub_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, buf.sub_rank_sums,
            buf.sub_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false, use_gmem);

        // Scatter to global output
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
                std::string("CUDA error in wilcoxon streaming: ") +
                cudaGetErrorString(err));
    }

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

/**
 * CSC-direct OVR streaming pipeline.
 *
 * Takes GPU CSC arrays directly — no format conversion needed.
 * For each sub-batch: extract dense columns from CSC → sort → rank.
 * CSC extraction is a simple scatter (no binary search), faster than CSR.
 */
static void ovr_streaming_csc_impl(
    const float* csc_data, const int* csc_indices, const int* csc_indptr,
    const int* group_codes, double* rank_sums, double* tie_corr, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, int sub_batch_cols) {
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
    struct StreamBuf {
        float* dense;
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
        bufs[s].dense = pool.alloc<float>(sub_items);
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

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // Zero dense buffer
        cudaMemsetAsync(buf.dense, 0, sb_items * sizeof(float), stream);

        // Extract dense columns from CSC — simple scatter, no binary search
        csc_extract_f32_kernel<<<sb_cols, 256, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, buf.dense, n_rows, col);

        // Fill segment offsets + row indices
        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.dense, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        if (use_gmem) {
            cudaMemsetAsync(buf.sub_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, buf.sub_rank_sums,
            buf.sub_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false, use_gmem);

        // Scatter to global output
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
                std::string("CUDA error in wilcoxon streaming: ") +
                cudaGetErrorString(err));
    }

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

/**
 * Host-streaming CSC OVR pipeline.
 *
 * CSC arrays live on host.  Only the sparse data for each sub-batch of
 * columns is transferred to GPU, so GPU memory is O(sub_batch * n_rows).
 * H2D of sub-batch N+1 overlaps compute of sub-batch N via multi-stream.
 */
static void ovr_streaming_csc_host_impl(
    const float* h_data, const int* h_indices, const int* h_indptr,
    const int* h_group_codes, double* h_rank_sums, double* h_tie_corr,
    int n_rows, int n_cols, int n_groups, bool compute_tie_corr,
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

    // Find max nnz across any sub-batch to size the sparse transfer buffers
    size_t max_nnz = 0;
    for (int col = 0; col < n_cols; col += sub_batch_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        size_t nnz = (size_t)(h_indptr[col + sb_cols] - h_indptr[col]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    // Allocate per-stream buffers via RMM pool
    RmmPool pool;
    int* d_group_codes = pool.alloc<int>(n_rows);
    struct StreamBuf {
        float* d_sparse_data;
        int* d_sparse_indices;
        int* d_indptr;
        float* dense;
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_sparse_data = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<int>(max_nnz);
        bufs[s].d_indptr = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].dense = pool.alloc<float>(sub_items);
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr = pool.alloc<double>(sub_batch_cols);
    }

    // Group codes on GPU (transferred once)
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);

    // Pin host memory for async transfers
    cudaHostRegister(const_cast<float*>(h_data),
                     (size_t)h_indptr[n_cols] * sizeof(float), 0);
    cudaHostRegister(const_cast<int*>(h_indices),
                     (size_t)h_indptr[n_cols] * sizeof(int), 0);
    cudaHostRegister(h_rank_sums, (size_t)n_groups * n_cols * sizeof(double),
                     0);
    cudaHostRegister(h_tie_corr, n_cols * sizeof(double), 0);

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // H2D: transfer sparse data for this column range
        int ptr_start = h_indptr[col];
        int ptr_end = h_indptr[col + sb_cols];
        size_t nnz = (size_t)(ptr_end - ptr_start);
        cudaMemcpyAsync(buf.d_sparse_data, h_data + ptr_start,
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buf.d_sparse_indices, h_indices + ptr_start,
                        nnz * sizeof(int), cudaMemcpyHostToDevice, stream);

        // Transfer adjusted indptr (rebased to 0)
        // h_indptr[col..col+sb_cols] - h_indptr[col]
        {
            std::vector<int> h_adj(sb_cols + 1);
            for (int i = 0; i <= sb_cols; i++)
                h_adj[i] = h_indptr[col + i] - ptr_start;
            cudaMemcpy(buf.d_indptr, h_adj.data(), (sb_cols + 1) * sizeof(int),
                       cudaMemcpyHostToDevice);
        }

        // Zero dense buffer
        cudaMemsetAsync(buf.dense, 0, sb_items * sizeof(float), stream);

        // CSC extract from transferred sparse data (col_start=0 because
        // indptr is rebased and data/indices are for this sub-batch only)
        csc_extract_f32_kernel<<<sb_cols, 256, 0, stream>>>(
            buf.d_sparse_data, buf.d_sparse_indices, buf.d_indptr, buf.dense,
            n_rows, 0);

        // Fill segment offsets + row indices
        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.dense, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        if (use_gmem) {
            cudaMemsetAsync(buf.d_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, d_group_codes, buf.d_rank_sums,
            buf.d_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false, use_gmem);

        // D2H: scatter results to host output
        cudaMemcpy2DAsync(h_rank_sums + col, n_cols * sizeof(double),
                          buf.d_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToHost, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(h_tie_corr + col, buf.d_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToHost,
                            stream);
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

    cudaHostUnregister(const_cast<float*>(h_data));
    cudaHostUnregister(const_cast<int*>(h_indices));
    cudaHostUnregister(h_rank_sums);
    cudaHostUnregister(h_tie_corr);

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

/**
 * Host-streaming dense OVR pipeline.
 *
 * Dense F-order float32 block lives on host.  Sub-batches of 64 columns
 * are transferred to GPU per stream, so GPU memory is O(sub_batch * n_rows).
 */
static void ovr_streaming_dense_host_impl(
    const float* h_block, const int* h_group_codes, double* h_rank_sums,
    double* h_tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
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
        float* d_block;
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_block = pool.alloc<float>(sub_items);
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr = pool.alloc<double>(sub_batch_cols);
    }

    // Group codes on GPU (transferred once)
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);

    // Pin host memory
    cudaHostRegister(const_cast<float*>(h_block),
                     (size_t)n_rows * n_cols * sizeof(float), 0);
    cudaHostRegister(h_rank_sums, (size_t)n_groups * n_cols * sizeof(double),
                     0);
    cudaHostRegister(h_tie_corr, n_cols * sizeof(double), 0);

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // H2D: column sub-batch (F-order → contiguous)
        cudaMemcpyAsync(buf.d_block, h_block + (long long)col * n_rows,
                        sb_items * sizeof(float), cudaMemcpyHostToDevice,
                        stream);

        // Fill segment offsets + row indices
        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.d_block, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        if (use_gmem) {
            cudaMemsetAsync(buf.d_rank_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, d_group_codes, buf.d_rank_sums,
            buf.d_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false, use_gmem);

        // D2H: scatter results
        cudaMemcpy2DAsync(h_rank_sums + col, n_cols * sizeof(double),
                          buf.d_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToHost, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(h_tie_corr + col, buf.d_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToHost,
                            stream);
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

    cudaHostUnregister(const_cast<float*>(h_block));
    cudaHostUnregister(h_rank_sums);
    cudaHostUnregister(h_tie_corr);

    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}

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
        "ovr_streaming_csr",
        [](gpu_array_c<const float, Device> csr_data,
           gpu_array_c<const int, Device> csr_indices,
           gpu_array_c<const int, Device> csr_indptr,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, int sub_batch_cols) {
            ovr_streaming_csr_impl(
                csr_data.data(), csr_indices.data(), csr_indptr.data(),
                group_codes.data(), rank_sums.data(), tie_corr.data(), n_rows,
                n_cols, n_groups, compute_tie_corr, sub_batch_cols);
        },
        "csr_data"_a, "csr_indices"_a, "csr_indptr"_a, "group_codes"_a,
        "rank_sums"_a, "tie_corr"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovr_streaming_csc",
        [](gpu_array_c<const float, Device> csc_data,
           gpu_array_c<const int, Device> csc_indices,
           gpu_array_c<const int, Device> csc_indptr,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, int sub_batch_cols) {
            ovr_streaming_csc_impl(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                group_codes.data(), rank_sums.data(), tie_corr.data(), n_rows,
                n_cols, n_groups, compute_tie_corr, sub_batch_cols);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "group_codes"_a,
        "rank_sums"_a, "tie_corr"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);
}

NB_MODULE(_wilcoxon_ovr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);

    m.def(
        "ovr_streaming_csc_host",
        [](host_array<const float> h_data, host_array<const int> h_indices,
           host_array<const int> h_indptr, host_array<const int> h_group_codes,
           host_array_2d<double> h_rank_sums, host_array<double> h_tie_corr,
           int n_rows, int n_cols, int n_groups, bool compute_tie_corr,
           int sub_batch_cols) {
            ovr_streaming_csc_host_impl(
                h_data.data(), h_indices.data(), h_indptr.data(),
                h_group_codes.data(), h_rank_sums.data(), h_tie_corr.data(),
                n_rows, n_cols, n_groups, compute_tie_corr, sub_batch_cols);
        },
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_group_codes"_a,
        "h_rank_sums"_a, "h_tie_corr"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovr_streaming_dense_host",
        [](host_array_2d<const float> h_block,
           host_array<const int> h_group_codes,
           host_array_2d<double> h_rank_sums, host_array<double> h_tie_corr,
           int n_rows, int n_cols, int n_groups, bool compute_tie_corr,
           int sub_batch_cols) {
            ovr_streaming_dense_host_impl(h_block.data(), h_group_codes.data(),
                                          h_rank_sums.data(), h_tie_corr.data(),
                                          n_rows, n_cols, n_groups,
                                          compute_tie_corr, sub_batch_cols);
        },
        "h_block"_a, "h_group_codes"_a, "h_rank_sums"_a, "h_tie_corr"_a,
        nb::kw_only(), "n_rows"_a, "n_cols"_a, "n_groups"_a,
        "compute_tie_corr"_a, "sub_batch_cols"_a = SUB_BATCH_COLS);
}
