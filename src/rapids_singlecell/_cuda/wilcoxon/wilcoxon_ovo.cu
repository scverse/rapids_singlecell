#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_common.cuh"
#include "kernels_wilcoxon_ovo.cuh"

using namespace nb::literals;

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
 * Extract specific rows from CSC into dense F-order, using a row lookup map.
 * row_map[original_row] = output_row_index (or -1 to skip).
 * One block per column, threads scatter matching nonzeros.
 * Output must be pre-zeroed.
 */
__global__ void csc_extract_mapped_kernel(const float* __restrict__ data,
                                          const int* __restrict__ indices,
                                          const int* __restrict__ indptr,
                                          const int* __restrict__ row_map,
                                          float* __restrict__ out, int n_target,
                                          int col_start) {
    int col_local = blockIdx.x;
    int col = col_start + col_local;

    int start = indptr[col];
    int end = indptr[col + 1];

    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        int out_row = row_map[indices[p]];
        if (out_row >= 0) {
            out[(long long)col_local * n_target + out_row] = data[p];
        }
    }
}

static size_t get_seg_sort_temp_bytes(int n_items, int n_segments) {
    size_t bytes = 0;
    auto* dk = reinterpret_cast<float*>(1);
    auto* doff = reinterpret_cast<int*>(1);
    cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes, dk, dk, n_items,
                                            n_segments, doff, doff + 1, 0, 32);
    return bytes;
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

    // Allocate per-stream buffers via RMM pool
    RmmPool pool;
    struct StreamBuf {
        float* grp_sorted;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
        bufs[s].seg_offsets = pool.alloc<int>(max_n_seg + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
    }

    // Compute max individual group size for accurate thread count
    std::vector<int> h_off(n_groups + 1);
    cudaMemcpy(h_off.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    int max_grp_size = 0;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_off[g + 1] - h_off[g];
        if (sz > max_grp_size) max_grp_size = sz;
    }
    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

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
    constexpr int TIER1_THRESHOLD = 2500;
    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    int max_grp_size = 0;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_offsets[g + 1] - h_offsets[g];
        if (sz > max_grp_size) max_grp_size = sz;
    }
    bool use_tier1 = (max_grp_size <= TIER1_THRESHOLD);

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

    if (!use_tier1) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg = n_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_grp_bytes, fk, fk, (int)sub_grp_items, max_grp_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    // Tier 1 precomputation
    int padded_grp_size = 0;
    int tier1_tpb = 0;
    size_t tier1_smem = 0;
    if (use_tier1) {
        padded_grp_size = 1;
        while (padded_grp_size < max_grp_size) padded_grp_size <<= 1;
        tier1_tpb = std::min(padded_grp_size, MAX_THREADS_PER_BLOCK);
        tier1_smem = padded_grp_size * sizeof(float) + 32 * sizeof(double);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    // Allocate per-stream buffers via RMM pool
    RmmPool pool;
    struct StreamBuf {
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        uint8_t* cub_temp;
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
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (!use_tier1) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_seg = n_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_seg + 1);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
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
        }

        if (use_tier1) {
            // ---- Tier 1: fused smem sort + binary search ----
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.sub_rank_sums,
                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size);
        } else {
            // ---- Tier 3: CUB segmented sort + binary search ----
            int sb_grp_seg = n_groups * sb_cols;
            {
                int total = sb_grp_seg + 1;
                int blk = (total + 255) / 256;
                build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                    grp_offsets, buf.grp_seg_offsets, n_all_grp, n_groups,
                    sb_cols);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_items_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, grp_offsets,
                    buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr);
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
    constexpr int TIER1_THRESHOLD = 2500;
    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    int max_grp_size = 0;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_offsets[g + 1] - h_offsets[g];
        if (sz > max_grp_size) max_grp_size = sz;
    }
    bool use_tier1 = (max_grp_size <= TIER1_THRESHOLD);

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
    if (!use_tier1) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg = n_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_grp_bytes, fk, fk, (int)sub_grp_items, max_grp_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    // Tier 1 precomputation
    int padded_grp_size = 0;
    int tier1_tpb = 0;
    size_t tier1_smem = 0;
    if (use_tier1) {
        padded_grp_size = 1;
        while (padded_grp_size < max_grp_size) padded_grp_size <<= 1;
        tier1_tpb = std::min(padded_grp_size, MAX_THREADS_PER_BLOCK);
        tier1_smem = padded_grp_size * sizeof(float) + 32 * sizeof(double);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmPool pool;
    struct StreamBuf {
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        uint8_t* cub_temp;
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
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (!use_tier1) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = n_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_grp_seg + 1);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
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
        csc_extract_mapped_kernel<<<sb_cols, 256, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, ref_row_map, buf.ref_dense,
            n_ref, col);
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
        csc_extract_mapped_kernel<<<sb_cols, 256, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, grp_row_map, buf.grp_dense,
            n_all_grp, col);

        if (use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.sub_rank_sums,
                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size);
        } else {
            int sb_grp_seg = n_groups * sb_cols;
            {
                int total = sb_grp_seg + 1;
                int blk = (total + 255) / 256;
                build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                    grp_offsets, buf.grp_seg_offsets, n_all_grp, n_groups,
                    sb_cols);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_items_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, grp_offsets,
                    buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr);
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
 * Host-streaming CSC OVO pipeline.
 *
 * CSC arrays live on host.  Only the sparse data for each sub-batch of
 * columns is transferred to GPU.  Row maps + group offsets are uploaded once.
 * Results are written back to host per sub-batch.
 */
static void ovo_streaming_csc_host_impl(
    const float* h_data, const int* h_indices, const int* h_indptr,
    const int* h_ref_row_map, const int* h_grp_row_map,
    const int* h_grp_offsets, double* h_rank_sums, double* h_tie_corr,
    int n_ref, int n_all_grp, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch from host offsets ----
    constexpr int TIER1_THRESHOLD = 2500;
    int max_grp_size = 0;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > max_grp_size) max_grp_size = sz;
    }
    bool use_tier1 = (max_grp_size <= TIER1_THRESHOLD);

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
    if (!use_tier1) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg = n_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_grp_bytes, fk, fk, (int)sub_grp_items, max_grp_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    int padded_grp_size = 0;
    int tier1_tpb = 0;
    size_t tier1_smem = 0;
    if (use_tier1) {
        padded_grp_size = 1;
        while (padded_grp_size < max_grp_size) padded_grp_size <<= 1;
        tier1_tpb = std::min(padded_grp_size, MAX_THREADS_PER_BLOCK);
        tier1_smem = padded_grp_size * sizeof(float) + 32 * sizeof(double);
    }

    // Max nnz across any sub-batch for sparse transfer buffer sizing
    size_t max_nnz = 0;
    for (int c = 0; c < n_cols; c += sub_batch_cols) {
        int sb = std::min(sub_batch_cols, n_cols - c);
        size_t nnz = (size_t)(h_indptr[c + sb] - h_indptr[c]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmPool pool;

    // GPU copies of row maps + group offsets (uploaded once)
    int* d_ref_row_map = pool.alloc<int>(n_rows);
    int* d_grp_row_map = pool.alloc<int>(n_rows);
    int* d_grp_offsets = pool.alloc<int>(n_groups + 1);
    cudaMemcpy(d_ref_row_map, h_ref_row_map, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_row_map, h_grp_row_map, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets, h_grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    struct StreamBuf {
        float* d_sparse_data;
        int* d_sparse_indices;
        int* d_indptr;
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_sparse_data = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<int>(max_nnz);
        bufs[s].d_indptr = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (!use_tier1) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = n_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_grp_seg + 1);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
        }
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    // Pin host memory for async transfers
    cudaHostRegister(const_cast<float*>(h_data),
                     (size_t)h_indptr[n_cols] * sizeof(float), 0);
    cudaHostRegister(const_cast<int*>(h_indices),
                     (size_t)h_indptr[n_cols] * sizeof(int), 0);
    cudaHostRegister(h_rank_sums, (size_t)n_groups * n_cols * sizeof(double),
                     0);
    cudaHostRegister(h_tie_corr, (size_t)n_groups * n_cols * sizeof(double), 0);

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_ref_actual = n_ref * sb_cols;
        int sb_grp_actual = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // ---- H2D: sparse data for this column range ----
        int ptr_start = h_indptr[col];
        int ptr_end = h_indptr[col + sb_cols];
        size_t nnz = (size_t)(ptr_end - ptr_start);
        cudaMemcpyAsync(buf.d_sparse_data, h_data + ptr_start,
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buf.d_sparse_indices, h_indices + ptr_start,
                        nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
        {
            std::vector<int> h_adj(sb_cols + 1);
            for (int i = 0; i <= sb_cols; i++)
                h_adj[i] = h_indptr[col + i] - ptr_start;
            cudaMemcpy(buf.d_indptr, h_adj.data(), (sb_cols + 1) * sizeof(int),
                       cudaMemcpyHostToDevice);
        }

        // ---- Extract ref from CSC via row_map, sort ----
        cudaMemsetAsync(buf.ref_dense, 0, sb_ref_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, 256, 0, stream>>>(
            buf.d_sparse_data, buf.d_sparse_indices, buf.d_indptr,
            d_ref_row_map, buf.ref_dense, n_ref, 0);
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortKeys(
                buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                sb_ref_actual, sb_cols, buf.ref_seg_offsets,
                buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // ---- Extract grp from CSC via row_map ----
        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, 256, 0, stream>>>(
            buf.d_sparse_data, buf.d_sparse_indices, buf.d_indptr,
            d_grp_row_map, buf.grp_dense, n_all_grp, 0);

        // ---- Tier dispatch: sort grp + rank ----
        if (use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, d_grp_offsets, buf.d_rank_sums,
                buf.d_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size);
        } else {
            int sb_grp_seg = n_groups * sb_cols;
            {
                int total = sb_grp_seg + 1;
                int blk = (total + 255) / 256;
                build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                    d_grp_offsets, buf.grp_seg_offsets, n_all_grp, n_groups,
                    sb_cols);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, d_grp_offsets,
                    buf.d_rank_sums, buf.d_tie_corr, n_ref, n_all_grp, sb_cols,
                    n_groups, compute_tie_corr);
            }
        }

        // ---- D2H: scatter results ----
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
 * Host CSR OVO pipeline — preload reference, stream perturbations.
 *
 * Two-phase approach:
 *   Phase 1: Transfer CSR to GPU, extract ref rows for ALL columns, sort once.
 *   Phase 2: For each column sub-batch, extract only grp rows, sort, rank
 *            against the pre-sorted reference.
 *
 * The reference is sorted once (not per sub-batch), saving ~50% of the
 * per-sub-batch extraction + sort work.
 */
static void ovo_streaming_csr_host_impl(
    const float* h_data, const int* h_indices, const int* h_indptr,
    const int* h_ref_row_ids, const int* h_grp_row_ids,
    const int* h_grp_offsets, double* h_rank_sums, double* h_tie_corr,
    int n_ref, int n_all_grp, int n_rows, int n_cols, int n_groups, int nnz,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch from host offsets ----
    constexpr int TIER1_THRESHOLD = 2500;
    int max_grp_size = 0;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > max_grp_size) max_grp_size = sz;
    }
    bool use_tier1 = (max_grp_size <= TIER1_THRESHOLD);

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;

    // CUB temp — sized for the larger of ref (full) or grp (sub-batch)
    size_t ref_total = (size_t)n_ref * n_cols;
    size_t cub_ref_bytes = 0;
    {
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, cub_ref_bytes, fk, fk,
                                                (int)ref_total, n_cols, doff,
                                                doff + 1, BEGIN_BIT, END_BIT);
    }
    size_t cub_temp_bytes = cub_ref_bytes;
    if (!use_tier1) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg = n_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_grp_bytes, fk, fk, (int)sub_grp_items, max_grp_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    int padded_grp_size = 0;
    int tier1_tpb = 0;
    size_t tier1_smem = 0;
    if (use_tier1) {
        padded_grp_size = 1;
        while (padded_grp_size < max_grp_size) padded_grp_size <<= 1;
        tier1_tpb = std::min(padded_grp_size, MAX_THREADS_PER_BLOCK);
        tier1_smem = padded_grp_size * sizeof(float) + 32 * sizeof(double);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmPool pool;

    // ---- Phase 1: Transfer CSR, extract + sort reference (all columns) ----
    float* d_data = pool.alloc<float>(nnz);
    int* d_indices = pool.alloc<int>(nnz);
    int* d_indptr = pool.alloc<int>(n_rows + 1);
    int* d_ref_row_ids = pool.alloc<int>(n_ref);
    int* d_grp_row_ids = pool.alloc<int>(n_all_grp);
    int* d_grp_offsets = pool.alloc<int>(n_groups + 1);

    cudaHostRegister(const_cast<float*>(h_data), (size_t)nnz * sizeof(float),
                     0);
    cudaHostRegister(const_cast<int*>(h_indices), (size_t)nnz * sizeof(int), 0);
    cudaMemcpyAsync(d_data, h_data, (size_t)nnz * sizeof(float),
                    cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_indices, h_indices, (size_t)nnz * sizeof(int),
                    cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy(d_indptr, h_indptr, (n_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_row_ids, h_ref_row_ids, n_ref * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_row_ids, h_grp_row_ids, n_all_grp * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets, h_grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaStreamSynchronize(streams[0]);

    // Extract ref for ALL columns, sort once
    float* ref_dense = pool.alloc<float>(ref_total);
    float* ref_sorted = pool.alloc<float>(ref_total);
    cudaMemset(ref_dense, 0, ref_total * sizeof(float));
    {
        int tpb = round_up_to_warp(n_ref);
        int blk = (n_ref + tpb - 1) / tpb;
        csr_extract_dense_kernel<<<blk, tpb>>>(d_data, d_indices, d_indptr,
                                               d_ref_row_ids, ref_dense, n_ref,
                                               0, n_cols);
    }
    {
        int* ref_seg = pool.alloc<int>(n_cols + 1);
        upload_linear_offsets(ref_seg, n_cols, n_ref, nullptr);
        uint8_t* cub_tmp = pool.alloc<uint8_t>(cub_ref_bytes);
        size_t temp = cub_ref_bytes;
        cub::DeviceSegmentedRadixSort::SortKeys(
            cub_tmp, temp, ref_dense, ref_sorted, (int)ref_total, n_cols,
            ref_seg, ref_seg + 1, BEGIN_BIT, END_BIT);
    }
    cudaDeviceSynchronize();

    // ---- Phase 2: Stream grp sub-batches, rank against pre-sorted ref ----
    struct StreamBuf {
        float* grp_dense;
        float* grp_sorted;
        int* grp_seg_offsets;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (!use_tier1) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = n_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_grp_seg + 1);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
        }
    }

    int tpb_extract = round_up_to_warp(n_all_grp);
    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    cudaHostRegister(h_rank_sums, (size_t)n_groups * n_cols * sizeof(double),
                     0);
    cudaHostRegister(h_tie_corr, (size_t)n_groups * n_cols * sizeof(double), 0);

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_grp_actual = n_all_grp * sb_cols;
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // Extract grp only (ref already sorted)
        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_actual * sizeof(float),
                        stream);
        {
            int blk = (n_all_grp + tpb_extract - 1) / tpb_extract;
            csr_extract_dense_kernel<<<blk, tpb_extract, 0, stream>>>(
                d_data, d_indices, d_indptr, d_grp_row_ids, buf.grp_dense,
                n_all_grp, col, col + sb_cols);
        }

        // Rank against pre-sorted ref (just slice into ref_sorted)
        const float* ref_sub = ref_sorted + (long long)col * n_ref;
        if (use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                ref_sub, buf.grp_dense, d_grp_offsets, buf.d_rank_sums,
                buf.d_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size);
        } else {
            int sb_grp_seg = n_groups * sb_cols;
            {
                int total = sb_grp_seg + 1;
                int blk = (total + 255) / 256;
                build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                    d_grp_offsets, buf.grp_seg_offsets, n_all_grp, n_groups,
                    sb_cols);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    ref_sub, buf.grp_sorted, d_grp_offsets, buf.d_rank_sums,
                    buf.d_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                    compute_tie_corr);
            }
        }

        // D2H results
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
static void ovo_streaming_dense_host_impl(
    const float* h_block, const int* h_ref_row_ids, const int* h_grp_row_ids,
    const int* h_grp_offsets, double* h_rank_sums, double* h_tie_corr,
    int n_ref, int n_all_grp, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // ---- Tier dispatch from host offsets ----
    constexpr int TIER1_THRESHOLD = 2500;
    int max_grp_size = 0;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > max_grp_size) max_grp_size = sz;
    }
    bool use_tier1 = (max_grp_size <= TIER1_THRESHOLD);

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
    if (!use_tier1) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg = n_groups * sub_batch_cols;
        auto* fk = reinterpret_cast<float*>(1);
        auto* doff = reinterpret_cast<int*>(1);
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, cub_grp_bytes, fk, fk, (int)sub_grp_items, max_grp_seg,
            doff, doff + 1, BEGIN_BIT, END_BIT);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    int padded_grp_size = 0;
    int tier1_tpb = 0;
    size_t tier1_smem = 0;
    if (use_tier1) {
        padded_grp_size = 1;
        while (padded_grp_size < max_grp_size) padded_grp_size <<= 1;
        tier1_tpb = std::min(padded_grp_size, MAX_THREADS_PER_BLOCK);
        tier1_smem = padded_grp_size * sizeof(float) + 32 * sizeof(double);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmPool pool;

    // GPU copies of row_ids + group offsets (uploaded once)
    int* d_ref_row_ids = pool.alloc<int>(n_ref);
    int* d_grp_row_ids = pool.alloc<int>(n_all_grp);
    int* d_grp_offsets = pool.alloc<int>(n_groups + 1);
    cudaMemcpy(d_ref_row_ids, h_ref_row_ids, n_ref * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_row_ids, h_grp_row_ids, n_all_grp * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets, h_grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    struct StreamBuf {
        float* d_block;  // H2D sub-batch (all rows)
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_block = pool.alloc<float>(sub_dense);
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (!use_tier1) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = n_groups * sub_batch_cols;
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_grp_seg + 1);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
        }
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    // Pin host memory
    cudaHostRegister(const_cast<float*>(h_block),
                     (size_t)n_rows * n_cols * sizeof(float), 0);
    cudaHostRegister(h_rank_sums, (size_t)n_groups * n_cols * sizeof(double),
                     0);
    cudaHostRegister(h_tie_corr, (size_t)n_groups * n_cols * sizeof(double), 0);

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

        // ---- H2D: dense column sub-batch (F-order, contiguous) ----
        cudaMemcpyAsync(buf.d_block, h_block + (long long)col * n_rows,
                        sb_dense * sizeof(float), cudaMemcpyHostToDevice,
                        stream);

        // ---- Gather ref rows, sort ----
        dense_gather_rows_kernel<<<sb_cols, 256, 0, stream>>>(
            buf.d_block, d_ref_row_ids, buf.ref_dense, n_rows, n_ref, sb_cols);
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        {
            size_t temp = cub_temp_bytes;
            cub::DeviceSegmentedRadixSort::SortKeys(
                buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                sb_ref_actual, sb_cols, buf.ref_seg_offsets,
                buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        // ---- Gather grp rows ----
        dense_gather_rows_kernel<<<sb_cols, 256, 0, stream>>>(
            buf.d_block, d_grp_row_ids, buf.grp_dense, n_rows, n_all_grp,
            sb_cols);

        // ---- Tier dispatch: sort grp + rank ----
        if (use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, d_grp_offsets, buf.d_rank_sums,
                buf.d_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size);
        } else {
            int sb_grp_seg = n_groups * sb_cols;
            {
                int total = sb_grp_seg + 1;
                int blk = (total + 255) / 256;
                build_seg_offsets_kernel<<<blk, 256, 0, stream>>>(
                    d_grp_offsets, buf.grp_seg_offsets, n_all_grp, n_groups,
                    sb_cols);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, d_grp_offsets,
                    buf.d_rank_sums, buf.d_tie_corr, n_ref, n_all_grp, sb_cols,
                    n_groups, compute_tie_corr);
            }
        }

        // ---- D2H: scatter results ----
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
}

NB_MODULE(_wilcoxon_ovo_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);

    m.def(
        "ovo_streaming_csc_host",
        [](host_array<const float> h_data, host_array<const int> h_indices,
           host_array<const int> h_indptr, host_array<const int> h_ref_row_map,
           host_array<const int> h_grp_row_map,
           host_array<const int> h_grp_offsets,
           host_array_2d<double> h_rank_sums, host_array_2d<double> h_tie_corr,
           int n_ref, int n_all_grp, int n_rows, int n_cols, int n_groups,
           bool compute_tie_corr, int sub_batch_cols) {
            ovo_streaming_csc_host_impl(
                h_data.data(), h_indices.data(), h_indptr.data(),
                h_ref_row_map.data(), h_grp_row_map.data(),
                h_grp_offsets.data(), h_rank_sums.data(), h_tie_corr.data(),
                n_ref, n_all_grp, n_rows, n_cols, n_groups, compute_tie_corr,
                sub_batch_cols);
        },
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_ref_row_map"_a,
        "h_grp_row_map"_a, "h_grp_offsets"_a, "h_rank_sums"_a, "h_tie_corr"_a,
        nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_rows"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovo_streaming_csr_host",
        [](host_array<const float> h_data, host_array<const int> h_indices,
           host_array<const int> h_indptr, host_array<const int> h_ref_row_ids,
           host_array<const int> h_grp_row_ids,
           host_array<const int> h_grp_offsets,
           host_array_2d<double> h_rank_sums, host_array_2d<double> h_tie_corr,
           int n_ref, int n_all_grp, int n_rows, int n_cols, int n_groups,
           int nnz, bool compute_tie_corr, int sub_batch_cols) {
            ovo_streaming_csr_host_impl(
                h_data.data(), h_indices.data(), h_indptr.data(),
                h_ref_row_ids.data(), h_grp_row_ids.data(),
                h_grp_offsets.data(), h_rank_sums.data(), h_tie_corr.data(),
                n_ref, n_all_grp, n_rows, n_cols, n_groups, nnz,
                compute_tie_corr, sub_batch_cols);
        },
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_ref_row_ids"_a,
        "h_grp_row_ids"_a, "h_grp_offsets"_a, "h_rank_sums"_a, "h_tie_corr"_a,
        nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_rows"_a, "n_cols"_a,
        "n_groups"_a, "nnz"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS);

    m.def(
        "ovo_streaming_dense_host",
        [](host_array_2d<const float> h_block,
           host_array<const int> h_ref_row_ids,
           host_array<const int> h_grp_row_ids,
           host_array<const int> h_grp_offsets,
           host_array_2d<double> h_rank_sums, host_array_2d<double> h_tie_corr,
           int n_ref, int n_all_grp, int n_rows, int n_cols, int n_groups,
           bool compute_tie_corr, int sub_batch_cols) {
            ovo_streaming_dense_host_impl(
                h_block.data(), h_ref_row_ids.data(), h_grp_row_ids.data(),
                h_grp_offsets.data(), h_rank_sums.data(), h_tie_corr.data(),
                n_ref, n_all_grp, n_rows, n_cols, n_groups, compute_tie_corr,
                sub_batch_cols);
        },
        "h_block"_a, "h_ref_row_ids"_a, "h_grp_row_ids"_a, "h_grp_offsets"_a,
        "h_rank_sums"_a, "h_tie_corr"_a, nb::kw_only(), "n_ref"_a,
        "n_all_grp"_a, "n_rows"_a, "n_cols"_a, "n_groups"_a,
        "compute_tie_corr"_a, "sub_batch_cols"_a = SUB_BATCH_COLS);
}
