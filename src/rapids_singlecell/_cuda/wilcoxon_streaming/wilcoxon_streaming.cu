#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "../wilcoxon/kernels_wilcoxon.cuh"
#include "../wilcoxon/kernels_wilcoxon_ovo.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int N_STREAMS = 4;
constexpr int SUB_BATCH_COLS = 32;
constexpr int BEGIN_BIT = 0;
constexpr int END_BIT = 32;

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

/**
 * Extract dense F-order float32 block from CSR.
 * All rows, column range [col_start, col_stop).
 * One thread per row, binary search for col_start.
 * Output must be pre-zeroed.
 */
__global__ void csr_extract_f32_kernel(const float* __restrict__ data,
                                       const int* __restrict__ indices,
                                       const int* __restrict__ indptr,
                                       float* __restrict__ out, int n_rows,
                                       int col_start, int col_stop) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int rs = indptr[row];
    int re = indptr[row + 1];

    // Binary search for col_start
    int lo = rs, hi = re;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (indices[m] < col_start)
            lo = m + 1;
        else
            hi = m;
    }

    int n_cols = col_stop - col_start;
    for (int p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        out[(long long)(c - col_start) * n_rows + row] = data[p];
    }
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

    // Per-stream buffers (allocated once, reused across sub-batches)
    struct StreamBuf {
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
    };

    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&bufs[s].keys_out, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].vals_in, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].vals_out, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].seg_offsets, (sub_batch_cols + 1) * sizeof(int));
        cudaMalloc(&bufs[s].cub_temp, cub_temp_bytes);
    }

    int tpb_rank = round_up_to_warp(n_rows);
    int smem_rank = (4 * n_groups + 32) * sizeof(double);

    // Allocate sub-batch output buffers per stream
    std::vector<double*> sub_rank_sums(n_streams);
    std::vector<double*> sub_tie_corr(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&sub_rank_sums[s],
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
        cudaMalloc(&sub_tie_corr[s], sub_batch_cols * sizeof(double));
    }

    // Process sub-batches round-robin across streams
    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // Fill segment offsets: [0, n_rows, 2*n_rows, ...]
        {
            int* h_off = new int[sb_cols + 1];
            for (int i = 0; i <= sb_cols; i++) h_off[i] = i * n_rows;
            cudaMemcpyAsync(buf.seg_offsets, h_off, (sb_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            delete[] h_off;
        }

        // Fill row indices
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
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, sub_rank_sums[s],
            sub_tie_corr[s], nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false);

        // Copy sub-batch results to global output (row-major scatter)
        // rank_sums is (n_groups, n_cols) row-major: group g, col c →
        // [g*n_cols+c] sub output is (n_groups, sb_cols): group g, local col lc
        // → [g*sb_cols+lc]
        cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                          sub_rank_sums[s], sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + col, sub_tie_corr[s],
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    // Sync all streams
    for (int s = 0; s < n_streams; s++) {
        cudaStreamSynchronize(streams[s]);
    }

    // Cleanup
    for (int s = 0; s < n_streams; s++) {
        cudaFree(bufs[s].keys_out);
        cudaFree(bufs[s].vals_in);
        cudaFree(bufs[s].vals_out);
        cudaFree(bufs[s].seg_offsets);
        cudaFree(bufs[s].cub_temp);
        cudaFree(sub_rank_sums[s]);
        cudaFree(sub_tie_corr[s]);
        cudaStreamDestroy(streams[s]);
    }
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

    struct StreamBuf {
        float* dense;  // extracted dense sub-batch
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
    };

    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&bufs[s].dense, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].keys_out, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].vals_in, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].vals_out, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].seg_offsets, (sub_batch_cols + 1) * sizeof(int));
        cudaMalloc(&bufs[s].cub_temp, cub_temp_bytes);
    }

    std::vector<double*> sub_rank_sums(n_streams);
    std::vector<double*> sub_tie_corr(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&sub_rank_sums[s],
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
        cudaMalloc(&sub_tie_corr[s], sub_batch_cols * sizeof(double));
    }

    int tpb_rank = round_up_to_warp(n_rows);
    int smem_rank = (4 * n_groups + 32) * sizeof(double);
    int tpb_extract = round_up_to_warp(n_rows);

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

        // Extract dense columns from CSR
        int extract_blocks = (n_rows + tpb_extract - 1) / tpb_extract;
        csr_extract_f32_kernel<<<extract_blocks, tpb_extract, 0, stream>>>(
            csr_data, csr_indices, csr_indptr, buf.dense, n_rows, col,
            col + sb_cols);

        // Fill segment offsets + row indices
        {
            int* h_off = new int[sb_cols + 1];
            for (int i = 0; i <= sb_cols; i++) h_off[i] = i * n_rows;
            cudaMemcpyAsync(buf.seg_offsets, h_off, (sb_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            delete[] h_off;
        }
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.dense, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, sub_rank_sums[s],
            sub_tie_corr[s], nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false);

        // Scatter to global output
        cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                          sub_rank_sums[s], sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + col, sub_tie_corr[s],
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    for (int s = 0; s < n_streams; s++) cudaStreamSynchronize(streams[s]);

    for (int s = 0; s < n_streams; s++) {
        cudaFree(bufs[s].dense);
        cudaFree(bufs[s].keys_out);
        cudaFree(bufs[s].vals_in);
        cudaFree(bufs[s].vals_out);
        cudaFree(bufs[s].seg_offsets);
        cudaFree(bufs[s].cub_temp);
        cudaFree(sub_rank_sums[s]);
        cudaFree(sub_tie_corr[s]);
        cudaStreamDestroy(streams[s]);
    }
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

    struct StreamBuf {
        float* dense;
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
    };

    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&bufs[s].dense, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].keys_out, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].vals_in, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].vals_out, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].seg_offsets, (sub_batch_cols + 1) * sizeof(int));
        cudaMalloc(&bufs[s].cub_temp, cub_temp_bytes);
    }

    std::vector<double*> sub_rank_sums(n_streams);
    std::vector<double*> sub_tie_corr(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&sub_rank_sums[s],
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
        cudaMalloc(&sub_tie_corr[s], sub_batch_cols * sizeof(double));
    }

    int tpb_rank = round_up_to_warp(n_rows);
    int smem_rank = (4 * n_groups + 32) * sizeof(double);

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
        {
            int* h_off = new int[sb_cols + 1];
            for (int i = 0; i <= sb_cols; i++) h_off[i] = i * n_rows;
            cudaMemcpyAsync(buf.seg_offsets, h_off, (sb_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            delete[] h_off;
        }
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.dense, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, sub_rank_sums[s],
            sub_tie_corr[s], nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false);

        // Scatter to global output
        cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                          sub_rank_sums[s], sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + col, sub_tie_corr[s],
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    for (int s = 0; s < n_streams; s++) cudaStreamSynchronize(streams[s]);

    for (int s = 0; s < n_streams; s++) {
        cudaFree(bufs[s].dense);
        cudaFree(bufs[s].keys_out);
        cudaFree(bufs[s].vals_in);
        cudaFree(bufs[s].vals_out);
        cudaFree(bufs[s].seg_offsets);
        cudaFree(bufs[s].cub_temp);
        cudaFree(sub_rank_sums[s]);
        cudaFree(sub_tie_corr[s]);
        cudaStreamDestroy(streams[s]);
    }
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

    // Group codes on GPU (transferred once)
    int* d_group_codes;
    cudaMalloc(&d_group_codes, n_rows * sizeof(int));
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);

    struct StreamBuf {
        float* d_sparse_data;   // H2D sparse values
        int* d_sparse_indices;  // H2D sparse row indices
        int* d_indptr;          // H2D indptr slice (sb_cols + 1)
        float* dense;           // extracted dense
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
        cudaMalloc(&bufs[s].d_sparse_data, max_nnz * sizeof(float));
        cudaMalloc(&bufs[s].d_sparse_indices, max_nnz * sizeof(int));
        cudaMalloc(&bufs[s].d_indptr, (sub_batch_cols + 1) * sizeof(int));
        cudaMalloc(&bufs[s].dense, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].keys_out, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].vals_in, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].vals_out, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].seg_offsets, (sub_batch_cols + 1) * sizeof(int));
        cudaMalloc(&bufs[s].cub_temp, cub_temp_bytes);
        cudaMalloc(&bufs[s].d_rank_sums,
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
        cudaMalloc(&bufs[s].d_tie_corr, sub_batch_cols * sizeof(double));
    }

    int tpb_rank = round_up_to_warp(n_rows);
    int smem_rank = (4 * n_groups + 32) * sizeof(double);

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
            int* h_adj = new int[sb_cols + 1];
            for (int i = 0; i <= sb_cols; i++)
                h_adj[i] = h_indptr[col + i] - ptr_start;
            cudaMemcpyAsync(buf.d_indptr, h_adj, (sb_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            delete[] h_adj;
        }

        // Zero dense buffer
        cudaMemsetAsync(buf.dense, 0, sb_items * sizeof(float), stream);

        // CSC extract from transferred sparse data (col_start=0 because
        // indptr is rebased and data/indices are for this sub-batch only)
        csc_extract_f32_kernel<<<sb_cols, 256, 0, stream>>>(
            buf.d_sparse_data, buf.d_sparse_indices, buf.d_indptr, buf.dense,
            n_rows, 0);

        // Fill segment offsets + row indices
        {
            int* h_off = new int[sb_cols + 1];
            for (int i = 0; i <= sb_cols; i++) h_off[i] = i * n_rows;
            cudaMemcpyAsync(buf.seg_offsets, h_off, (sb_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            delete[] h_off;
        }
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.dense, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, d_group_codes, buf.d_rank_sums,
            buf.d_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false);

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

    for (int s = 0; s < n_streams; s++) cudaStreamSynchronize(streams[s]);

    cudaHostUnregister(const_cast<float*>(h_data));
    cudaHostUnregister(const_cast<int*>(h_indices));
    cudaHostUnregister(h_rank_sums);
    cudaHostUnregister(h_tie_corr);

    cudaFree(d_group_codes);
    for (int s = 0; s < n_streams; s++) {
        cudaFree(bufs[s].d_sparse_data);
        cudaFree(bufs[s].d_sparse_indices);
        cudaFree(bufs[s].d_indptr);
        cudaFree(bufs[s].dense);
        cudaFree(bufs[s].keys_out);
        cudaFree(bufs[s].vals_in);
        cudaFree(bufs[s].vals_out);
        cudaFree(bufs[s].seg_offsets);
        cudaFree(bufs[s].cub_temp);
        cudaFree(bufs[s].d_rank_sums);
        cudaFree(bufs[s].d_tie_corr);
        cudaStreamDestroy(streams[s]);
    }
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

    int* d_group_codes;
    cudaMalloc(&d_group_codes, n_rows * sizeof(int));
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);

    struct StreamBuf {
        float* d_block;  // H2D dense sub-batch
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
        cudaMalloc(&bufs[s].d_block, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].keys_out, sub_items * sizeof(float));
        cudaMalloc(&bufs[s].vals_in, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].vals_out, sub_items * sizeof(int));
        cudaMalloc(&bufs[s].seg_offsets, (sub_batch_cols + 1) * sizeof(int));
        cudaMalloc(&bufs[s].cub_temp, cub_temp_bytes);
        cudaMalloc(&bufs[s].d_rank_sums,
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
        cudaMalloc(&bufs[s].d_tie_corr, sub_batch_cols * sizeof(double));
    }

    int tpb_rank = round_up_to_warp(n_rows);
    int smem_rank = (4 * n_groups + 32) * sizeof(double);

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
        {
            int* h_off = new int[sb_cols + 1];
            for (int i = 0; i <= sb_cols; i++) h_off[i] = i * n_rows;
            cudaMemcpyAsync(buf.seg_offsets, h_off, (sb_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            delete[] h_off;
        }
        fill_row_indices_kernel<<<sb_cols, 256, 0, stream>>>(buf.vals_in,
                                                             n_rows, sb_cols);

        // Sort
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            buf.cub_temp, temp, buf.d_block, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, BEGIN_BIT, END_BIT, stream);

        // Fused rank sums
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, d_group_codes, buf.d_rank_sums,
            buf.d_tie_corr, nullptr, nullptr, nullptr, n_rows, sb_cols,
            n_groups, compute_tie_corr, false);

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

    for (int s = 0; s < n_streams; s++) cudaStreamSynchronize(streams[s]);

    cudaHostUnregister(const_cast<float*>(h_block));
    cudaHostUnregister(h_rank_sums);
    cudaHostUnregister(h_tie_corr);

    cudaFree(d_group_codes);
    for (int s = 0; s < n_streams; s++) {
        cudaFree(bufs[s].d_block);
        cudaFree(bufs[s].keys_out);
        cudaFree(bufs[s].vals_in);
        cudaFree(bufs[s].vals_out);
        cudaFree(bufs[s].seg_offsets);
        cudaFree(bufs[s].cub_temp);
        cudaFree(bufs[s].d_rank_sums);
        cudaFree(bufs[s].d_tie_corr);
        cudaStreamDestroy(streams[s]);
    }
}

/**
 * Build segment offsets for CUB segmented sort of group data within a
 * sub-batch.  offset[c * n_groups + g] = c * n_all_grp + grp_offsets[g].
 * One thread per entry.
 */
__global__ void build_seg_offsets_kernel(
    const int* __restrict__ grp_offsets,  // (n_groups + 1,)
    int* __restrict__ out,                // (sb_cols * n_groups + 1,)
    int n_all_grp, int n_groups, int sb_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sb_cols * n_groups + 1;
    if (idx >= total) return;
    if (idx == sb_cols * n_groups) {
        out[idx] = sb_cols * n_all_grp;
    } else {
        int c = idx / n_groups;
        int g = idx % n_groups;
        out[idx] = c * n_all_grp + grp_offsets[g];
    }
}

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

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    int max_n_seg = n_groups * sub_batch_cols;
    size_t cub_temp_bytes = 0;
    {
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, cub_temp_bytes, fk, fk,
                                                (int)sub_grp_items, max_n_seg,
                                                doff, doff + 1, 0, 32);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    struct StreamBuf {
        float* grp_sorted;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };

    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        cudaMalloc(&bufs[s].grp_sorted, sub_grp_items * sizeof(float));
        cudaMalloc(&bufs[s].seg_offsets, (max_n_seg + 1) * sizeof(int));
        cudaMalloc(&bufs[s].cub_temp, cub_temp_bytes);
        cudaMalloc(&bufs[s].sub_rank_sums,
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
        cudaMalloc(&bufs[s].sub_tie_corr,
                   (size_t)n_groups * sub_batch_cols * sizeof(double));
    }

    // Import the presorted kernel from the OVO header
    // (included via kernels_wilcoxon_ovo.cuh)
    int tpb_rank = round_up_to_warp(std::min(n_all_grp, MAX_THREADS_PER_BLOCK));

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_n_seg = n_groups * sb_cols;
        int sb_grp_items = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // Build segment offsets on device
        {
            int total = sb_n_seg + 1;
            int blk = (total + 255) / 256;
            build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                grp_offsets, buf.seg_offsets, n_all_grp, n_groups, sb_cols);
        }

        // Sort group data for this sub-batch
        const float* grp_in = grp_data + (long long)col * n_all_grp;
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortKeys(
            buf.cub_temp, temp, grp_in, buf.grp_sorted, sb_grp_items, sb_n_seg,
            buf.seg_offsets, buf.seg_offsets + 1, 0, 32, stream);

        // Rank sums: binary search sorted ref for each group element
        const float* ref_sub = ref_sorted + (long long)col * n_ref;
        dim3 grid(sb_cols, n_groups);
        batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0, stream>>>(
            ref_sub, buf.grp_sorted, grp_offsets, buf.sub_rank_sums,
            buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
            compute_tie_corr);

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

    for (int s = 0; s < n_streams; s++) cudaStreamSynchronize(streams[s]);

    for (int s = 0; s < n_streams; s++) {
        cudaFree(bufs[s].grp_sorted);
        cudaFree(bufs[s].seg_offsets);
        cudaFree(bufs[s].cub_temp);
        cudaFree(bufs[s].sub_rank_sums);
        cudaFree(bufs[s].sub_tie_corr);
        cudaStreamDestroy(streams[s]);
    }
}

/**
 * Multi-GPU OVO streaming pipeline with host data.
 *
 * Ref block is sorted on GPU 0, then P2P copied to other GPUs.
 * Group data is streamed from host to each GPU's streams.
 */
static void ovo_streaming_multigpu_impl(
    const float* h_ref_sorted, const float* h_grp_data,
    const int* h_grp_offsets, double* h_rank_sums, double* h_tie_corr,
    int n_ref, int n_all_grp, int n_cols, int n_groups, bool compute_tie_corr,
    int sub_batch_cols, const int* h_device_ids, int n_devices) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    constexpr int STREAMS_PER_GPU = 2;

    // CUB temp for segmented sort of group data
    int max_n_seg = n_groups * sub_batch_cols;
    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    size_t cub_temp_bytes = 0;
    {
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, cub_temp_bytes, fk, fk,
                                                (int)sub_grp_items, max_n_seg,
                                                doff, doff + 1, 0, 32);
    }

    int tpb = round_up_to_warp(std::min(n_all_grp, MAX_THREADS_PER_BLOCK));

    struct GpuCtx {
        int device_id;
        cudaStream_t streams[2];
        float* d_ref_sorted;  // full ref, copied once
        int* d_grp_offsets;
        struct {
            float* d_grp_data;
            float* d_grp_sorted;
            int* d_seg_offsets;
            uint8_t* d_cub_temp;
            double* d_rank_sums;
            double* d_tie_corr;
        } buf[2];
    };

    std::vector<GpuCtx> gpus(n_devices);

    // Phase 1: allocate + upload ref + offsets to each GPU
    for (int d = 0; d < n_devices; d++) {
        auto& g = gpus[d];
        g.device_id = h_device_ids[d];
        cudaSetDevice(g.device_id);

        for (int s = 0; s < STREAMS_PER_GPU; s++)
            cudaStreamCreate(&g.streams[s]);

        size_t ref_size = (size_t)n_ref * n_cols;
        cudaMalloc(&g.d_ref_sorted, ref_size * sizeof(float));
        cudaMemcpyAsync(g.d_ref_sorted, h_ref_sorted, ref_size * sizeof(float),
                        cudaMemcpyHostToDevice, g.streams[0]);

        cudaMalloc(&g.d_grp_offsets, (n_groups + 1) * sizeof(int));
        cudaMemcpyAsync(g.d_grp_offsets, h_grp_offsets,
                        (n_groups + 1) * sizeof(int), cudaMemcpyHostToDevice,
                        g.streams[0]);

        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            cudaMalloc(&g.buf[s].d_grp_data, sub_grp_items * sizeof(float));
            cudaMalloc(&g.buf[s].d_grp_sorted, sub_grp_items * sizeof(float));
            cudaMalloc(&g.buf[s].d_seg_offsets, (max_n_seg + 1) * sizeof(int));
            cudaMalloc(&g.buf[s].d_cub_temp, cub_temp_bytes);
            cudaMalloc(&g.buf[s].d_rank_sums,
                       (size_t)n_groups * sub_batch_cols * sizeof(double));
            cudaMalloc(&g.buf[s].d_tie_corr,
                       (size_t)n_groups * sub_batch_cols * sizeof(double));
        }
    }

    // Phase 2: process sub-batches
    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_n_seg = n_groups * sb_cols;
        int sb_grp_items = n_all_grp * sb_cols;

        int d = (batch_idx / STREAMS_PER_GPU) % n_devices;
        int s = batch_idx % STREAMS_PER_GPU;
        auto& g = gpus[d];
        auto stream = g.streams[s];
        auto& buf = g.buf[s];

        cudaSetDevice(g.device_id);

        // H2D: group data sub-batch
        cudaMemcpyAsync(buf.d_grp_data, h_grp_data + (long long)col * n_all_grp,
                        sb_grp_items * sizeof(float), cudaMemcpyHostToDevice,
                        stream);

        // Build segment offsets on device
        {
            int total = sb_n_seg + 1;
            int blk = (total + 255) / 256;
            build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                g.d_grp_offsets, buf.d_seg_offsets, n_all_grp, n_groups,
                sb_cols);
        }

        // Sort group data
        size_t temp = cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortKeys(
            buf.d_cub_temp, temp, buf.d_grp_data, buf.d_grp_sorted,
            sb_grp_items, sb_n_seg, buf.d_seg_offsets, buf.d_seg_offsets + 1, 0,
            32, stream);

        // Rank sums
        const float* ref_sub = g.d_ref_sorted + (long long)col * n_ref;
        dim3 grid(sb_cols, n_groups);
        batched_rank_sums_presorted_kernel<<<grid, tpb, 0, stream>>>(
            ref_sub, buf.d_grp_sorted, g.d_grp_offsets, buf.d_rank_sums,
            buf.d_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
            compute_tie_corr);

        // D2H: scatter results
        cudaMemcpy2DAsync(h_rank_sums + col, n_cols * sizeof(double),
                          buf.d_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToHost, stream);
        if (compute_tie_corr) {
            cudaMemcpy2DAsync(h_tie_corr + col, n_cols * sizeof(double),
                              buf.d_tie_corr, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToHost, stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    // Phase 3: sync + cleanup
    for (int d = 0; d < n_devices; d++) {
        cudaSetDevice(gpus[d].device_id);
        for (int s = 0; s < STREAMS_PER_GPU; s++)
            cudaStreamSynchronize(gpus[d].streams[s]);
    }
    for (int d = 0; d < n_devices; d++) {
        cudaSetDevice(gpus[d].device_id);
        cudaFree(gpus[d].d_ref_sorted);
        cudaFree(gpus[d].d_grp_offsets);
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            cudaFree(gpus[d].buf[s].d_grp_data);
            cudaFree(gpus[d].buf[s].d_grp_sorted);
            cudaFree(gpus[d].buf[s].d_seg_offsets);
            cudaFree(gpus[d].buf[s].d_cub_temp);
            cudaFree(gpus[d].buf[s].d_rank_sums);
            cudaFree(gpus[d].buf[s].d_tie_corr);
            cudaStreamDestroy(gpus[d].streams[s]);
        }
    }
    cudaSetDevice(h_device_ids[0]);
}

// ============================================================================
// Nanobind module
// ============================================================================

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "Streaming Wilcoxon pipeline with multi-stream overlap";

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
}

NB_MODULE(_wilcoxon_streaming_cuda, m) {
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

    m.def(
        "ovo_streaming_multigpu",
        [](host_array_2d<const float> h_ref_sorted,
           host_array_2d<const float> h_grp_data,
           host_array<const int> h_grp_offsets,
           host_array_2d<double> h_rank_sums, host_array_2d<double> h_tie_corr,
           int n_ref, int n_all_grp, int n_cols, int n_groups,
           bool compute_tie_corr, int sub_batch_cols,
           host_array<const int> device_ids) {
            // Pin host arrays
            size_t ref_bytes = (size_t)n_ref * n_cols * sizeof(float);
            size_t grp_bytes = (size_t)n_all_grp * n_cols * sizeof(float);
            size_t rs_bytes = (size_t)n_groups * n_cols * sizeof(double);
            cudaHostRegister(const_cast<float*>(h_ref_sorted.data()), ref_bytes,
                             0);
            cudaHostRegister(const_cast<float*>(h_grp_data.data()), grp_bytes,
                             0);
            cudaHostRegister(const_cast<int*>(h_grp_offsets.data()),
                             (n_groups + 1) * sizeof(int), 0);
            cudaHostRegister(h_rank_sums.data(), rs_bytes, 0);
            cudaHostRegister(h_tie_corr.data(),
                             (size_t)n_groups * n_cols * sizeof(double), 0);

            ovo_streaming_multigpu_impl(
                h_ref_sorted.data(), h_grp_data.data(), h_grp_offsets.data(),
                h_rank_sums.data(), h_tie_corr.data(), n_ref, n_all_grp, n_cols,
                n_groups, compute_tie_corr, sub_batch_cols, device_ids.data(),
                static_cast<int>(device_ids.size()));

            cudaHostUnregister(const_cast<float*>(h_ref_sorted.data()));
            cudaHostUnregister(const_cast<float*>(h_grp_data.data()));
            cudaHostUnregister(const_cast<int*>(h_grp_offsets.data()));
            cudaHostUnregister(h_rank_sums.data());
            cudaHostUnregister(h_tie_corr.data());
        },
        "h_ref_sorted"_a, "h_grp_data"_a, "h_grp_offsets"_a, "h_rank_sums"_a,
        "h_tie_corr"_a, nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a, "sub_batch_cols"_a = SUB_BATCH_COLS,
        "device_ids"_a);
}
