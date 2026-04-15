#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_common.cuh"
#include "kernels_wilcoxon.cuh"

using namespace nb::literals;

/** Rebase a slice of indptr: out[i] = indptr[col + i] - indptr[col]. */
__global__ void rebase_indptr_kernel(const int* __restrict__ indptr,
                                     int* __restrict__ out, int col,
                                     int count) {
    int i = threadIdx.x;
    if (i < count) out[i] = indptr[col + i] - indptr[col];
}

/** Subtract a constant from an int array in-place. */
__global__ void subtract_scalar_kernel(int* __restrict__ data, int base,
                                       int count) {
    int i = threadIdx.x;
    if (i < count) data[i] -= base;
}

/** Count nonzeros per column from CSR. One thread per row. */
__global__ void csr_col_histogram_kernel(const int* __restrict__ indices,
                                         const int* __restrict__ indptr,
                                         int* __restrict__ col_counts,
                                         int n_rows, int n_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int rs = indptr[row];
    int re = indptr[row + 1];
    for (int p = rs; p < re; ++p) {
        int c = indices[p];
        if (c < n_cols) atomicAdd(&col_counts[c], 1);
    }
}

/**
 * Scatter CSR nonzeros into CSC layout for columns [col_start, col_stop).
 * write_pos[c - col_start] must be initialized to the prefix-sum offset
 * for column c.  Each thread atomically claims a unique destination slot.
 */
__global__ void csr_scatter_to_csc_kernel(
    const float* __restrict__ data, const int* __restrict__ indices,
    const int* __restrict__ indptr, int* __restrict__ write_pos,
    float* __restrict__ csc_vals, int* __restrict__ csc_row_idx, int n_rows,
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
    for (int p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        int dest = atomicAdd(&write_pos[c - col_start], 1);
        csc_vals[dest] = data[p];
        csc_row_idx[dest] = row;
    }
}

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
 * Sparse-aware host-streaming CSC OVR pipeline.
 *
 * Like ovr_streaming_csc_host_impl but sorts only stored nonzeros per column
 * instead of extracting dense blocks.  GPU memory is O(max_batch_nnz) instead
 * of O(sub_batch * n_rows), and sort work is proportional to nnz, not n_rows.
 */
static void ovr_sparse_csc_host_streaming_impl(
    const float* h_data, const int* h_indices, const int* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* h_rank_sums,
    double* h_tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
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
        float* d_sparse_data;
        int* d_sparse_indices;
        int* d_seg_offsets;
        float* keys_out;
        int* vals_out;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_sparse_data = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<int>(max_nnz);
        bufs[s].d_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].keys_out = pool.alloc<float>(max_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_nnz);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr = pool.alloc<double>(sub_batch_cols);
    }

    // Transfer group codes + sizes once
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    int tpb = 256;
    size_t smem_bytes = (size_t)(2 * n_groups + 32) * sizeof(double);

    // Pin host memory for async transfers
    cudaHostRegister(const_cast<float*>(h_data),
                     (size_t)h_indptr[n_cols] * sizeof(float), 0);
    cudaHostRegister(const_cast<int*>(h_indices),
                     (size_t)h_indptr[n_cols] * sizeof(int), 0);
    cudaHostRegister(const_cast<int*>(h_indptr),
                     (size_t)(n_cols + 1) * sizeof(int), 0);
    cudaHostRegister(h_rank_sums, (size_t)n_groups * n_cols * sizeof(double),
                     0);
    cudaHostRegister(h_tie_corr, n_cols * sizeof(double), 0);

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

        // H2D: transfer sparse data for this column range
        if (batch_nnz > 0) {
            cudaMemcpyAsync(buf.d_sparse_data, h_data + ptr_start,
                            (size_t)batch_nnz * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(buf.d_sparse_indices, h_indices + ptr_start,
                            (size_t)batch_nnz * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }

        // Async transfer indptr slice, then rebase on GPU
        cudaMemcpyAsync(buf.d_seg_offsets, h_indptr + col,
                        (sb_cols + 1) * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        subtract_scalar_kernel<<<1, sb_cols + 1, 0, stream>>>(
            buf.d_seg_offsets, ptr_start, sb_cols + 1);

        // CUB sort only stored nonzeros
        if (batch_nnz > 0) {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortPairs(
                buf.cub_temp, temp, buf.d_sparse_data, buf.keys_out,
                buf.d_sparse_indices, buf.vals_out, batch_nnz, sb_cols,
                buf.d_seg_offsets, buf.d_seg_offsets + 1, BEGIN_BIT, END_BIT,
                stream);
        }

        // Sparse rank kernel
        rank_sums_sparse_ovr_kernel<<<sb_cols, tpb, smem_bytes, stream>>>(
            buf.keys_out, buf.vals_out, buf.d_seg_offsets, d_group_codes,
            d_group_sizes, buf.d_rank_sums, buf.d_tie_corr, n_rows, sb_cols,
            n_groups, compute_tie_corr);

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
                std::string("CUDA error in sparse host CSC streaming: ") +
                cudaGetErrorString(err));
    }

    cudaHostUnregister(const_cast<float*>(h_data));
    cudaHostUnregister(const_cast<int*>(h_indices));
    cudaHostUnregister(const_cast<int*>(h_indptr));
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

    RmmPool pool;
    struct StreamBuf {
        float* keys_out;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
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
    }

    int tpb = 256;
    size_t smem_bytes = (size_t)(2 * n_groups + 32) * sizeof(double);

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
        rebase_indptr_kernel<<<1, sb_cols + 1, 0, stream>>>(
            csc_indptr, buf.seg_offsets, col, sb_cols + 1);

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
        rank_sums_sparse_ovr_kernel<<<sb_cols, tpb, smem_bytes, stream>>>(
            buf.keys_out, buf.vals_out, buf.seg_offsets, group_codes,
            group_sizes, buf.sub_rank_sums, buf.sub_tie_corr, n_rows, sb_cols,
            n_groups, compute_tie_corr);

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
        int tpb = 256;
        int blocks = (n_rows + tpb - 1) / tpb;
        csr_col_histogram_kernel<<<blocks, tpb>>>(csr_indices, csr_indptr,
                                                  d_col_counts, n_rows, n_cols);
    }
    std::vector<int> h_col_counts(n_cols);
    cudaMemcpy(h_col_counts.data(), d_col_counts, n_cols * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Per-batch prefix sums on host
    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    size_t max_batch_nnz = 0;
    size_t total_nnz = 0;

    // Flat array: n_batches × (sub_batch_cols + 1) offsets
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    std::vector<int> h_batch_nnz(n_batches);

    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int sb_cols = std::min(sub_batch_cols, n_cols - col_start);
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        off[0] = 0;
        for (int i = 0; i < sb_cols; i++)
            off[i + 1] = off[i] + h_col_counts[col_start + i];
        h_batch_nnz[b] = off[sb_cols];
        total_nnz += h_batch_nnz[b];
        if ((size_t)h_batch_nnz[b] > max_batch_nnz)
            max_batch_nnz = h_batch_nnz[b];
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

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

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
    }

    int tpb = 256;
    size_t smem_bytes = (size_t)(2 * n_groups + 32) * sizeof(double);
    int scatter_blocks = (n_rows + tpb - 1) / tpb;

    cudaDeviceSynchronize();

    // ---- Phase 2: Stream loop ----
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = b % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];
        int batch_nnz = h_batch_nnz[b];

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

            // CUB sort only the nonzeros
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortPairs(
                buf.cub_temp, temp, buf.csc_vals, buf.keys_out, buf.csc_row_idx,
                buf.vals_out, batch_nnz, sb_cols, buf.col_offsets,
                buf.col_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // Sparse rank kernel (handles implicit zeros analytically)
        rank_sums_sparse_ovr_kernel<<<sb_cols, tpb, smem_bytes, stream>>>(
            buf.keys_out, buf.vals_out, buf.col_offsets, group_codes,
            group_sizes, buf.sub_rank_sums, buf.sub_tie_corr, n_rows, sb_cols,
            n_groups, compute_tie_corr);

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
}

NB_MODULE(_wilcoxon_ovr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);

    m.def(
        "ovr_sparse_csc_host",
        [](host_array<const float> h_data, host_array<const int> h_indices,
           host_array<const int> h_indptr, host_array<const int> h_group_codes,
           host_array<double> h_group_sizes, host_array_2d<double> h_rank_sums,
           host_array<double> h_tie_corr, int n_rows, int n_cols, int n_groups,
           bool compute_tie_corr, int sub_batch_cols) {
            ovr_sparse_csc_host_streaming_impl(
                h_data.data(), h_indices.data(), h_indptr.data(),
                h_group_codes.data(), h_group_sizes.data(), h_rank_sums.data(),
                h_tie_corr.data(), n_rows, n_cols, n_groups, compute_tie_corr,
                sub_batch_cols);
        },
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_group_codes"_a,
        "h_group_sizes"_a, "h_rank_sums"_a, "h_tie_corr"_a, nb::kw_only(),
        "n_rows"_a, "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,
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
