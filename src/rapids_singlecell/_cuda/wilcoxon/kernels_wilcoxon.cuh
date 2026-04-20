#pragma once

#include <cuda_runtime.h>

/**
 * Fused rank-sum kernel: walk sorted data, compute per-group rank sums
 * and tie correction without materializing a rank matrix.
 *
 * Each thread processes a CONTIGUOUS chunk of sorted elements, detecting
 * tie groups by adjacent comparison (sequential access, no binary search).
 * Cross-boundary ties are resolved via binary search at chunk boundaries.
 *
 * When use_gmem is false, per-group accumulators live in shared memory
 * (fast atomics, limited to ~1500 groups on 48 KB devices).  When use_gmem
 * is true, accumulators write directly to ``rank_sums`` in global memory,
 * supporting an arbitrary number of groups.  The caller must pre-zero
 * ``rank_sums`` before launching in the gmem path.
 *
 * Shared memory layout:
 *   use_gmem=false: (n_groups + 32) doubles   (accumulators + warp buf)
 *   use_gmem=true:  32 doubles                 (warp buf only)
 */
__global__ void rank_sums_from_sorted_kernel(
    const float* __restrict__ sorted_vals,
    const int* __restrict__ sorted_row_idx, const int* __restrict__ group_codes,
    double* __restrict__ rank_sums, double* __restrict__ tie_corr, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, bool use_gmem) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    extern __shared__ double smem[];

    double* grp_sums;
    if (use_gmem) {
        // Global memory path: write directly to output (must be pre-zeroed)
        grp_sums = rank_sums + (size_t)col;  // stride: n_cols
    } else {
        // Shared memory path: per-block accumulators
        grp_sums = smem;
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            grp_sums[g] = 0.0;
        }
        __syncthreads();
    }

    const float* sv = sorted_vals + (size_t)col * n_rows;
    const int* si = sorted_row_idx + (size_t)col * n_rows;

    int chunk = (n_rows + blockDim.x - 1) / blockDim.x;
    int my_start = threadIdx.x * chunk;
    int my_end = my_start + chunk;
    if (my_end > n_rows) my_end = n_rows;

    double local_tie_sum = 0.0;

    // Stride for accumulator indexing: 1 for shared mem, n_cols for global mem
    int acc_stride = use_gmem ? n_cols : 1;

    int i = my_start;
    while (i < my_end) {
        double val = sv[i];

        int tie_local_end = i + 1;
        while (tie_local_end < my_end && sv[tie_local_end] == val)
            ++tie_local_end;

        int tie_global_start = i;
        if (i == my_start && i > 0 && sv[i - 1] == val) {
            int lo = 0, hi = i;
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (sv[mid] < val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            tie_global_start = lo;
        }

        int tie_global_end = tie_local_end;
        if (tie_local_end == my_end && tie_local_end < n_rows &&
            sv[tie_local_end] == val) {
            int lo = tie_local_end, hi = n_rows - 1;
            while (lo < hi) {
                int mid = hi - ((hi - lo) >> 1);
                if (sv[mid] > val)
                    hi = mid - 1;
                else
                    lo = mid;
            }
            tie_global_end = lo + 1;
        }

        int total_tie = tie_global_end - tie_global_start;
        double avg_rank = (double)(tie_global_start + tie_global_end + 1) / 2.0;

        for (int j = i; j < tie_local_end; ++j) {
            int grp = group_codes[si[j]];
            if (grp < n_groups) {
                atomicAdd(&grp_sums[grp * acc_stride], avg_rank);
            }
        }

        if (compute_tie_corr && tie_global_start >= my_start && total_tie > 1) {
            double t = (double)total_tie;
            local_tie_sum += t * t * t - t;
        }

        i = tie_local_end;
    }

    __syncthreads();

    // Copy shared memory accumulators to global output (smem path only)
    if (!use_gmem) {
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            rank_sums[(size_t)g * n_cols + col] = grp_sums[g];
        }
    }

    if (compute_tie_corr) {
        // Warp buf sits after accumulator array in shared memory.
        // gmem path: warp buf starts at smem[0].
        // smem path: n_groups doubles, then warp buf.
        int warp_buf_off = use_gmem ? 0 : n_groups;
        double* warp_buf = smem + warp_buf_off;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local_tie_sum += __shfl_down_sync(0xffffffff, local_tie_sum, off);
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        if (lane == 0) warp_buf[wid] = local_tie_sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            double val = (threadIdx.x < ((blockDim.x + 31) >> 5))
                             ? warp_buf[threadIdx.x]
                             : 0.0;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                val += __shfl_down_sync(0xffffffff, val, off);
            if (threadIdx.x == 0) {
                double n = (double)n_rows;
                double denom = n * n * n - n;
                tie_corr[col] = (denom > 0.0) ? (1.0 - val / denom) : 1.0;
            }
        }
    }
}

/**
 * Sparse-aware OVR rank-sum kernel for sorted stored values.
 *
 * After CUB sort the stored values are in ascending order:
 *   [negatives..., stored_zeros..., positives...]
 * Implicit zeros (n_rows − nnz_stored) are inserted analytically
 * between negatives and positives to form the full ranking.
 *
 * Full sorted array (conceptual):
 *   [negatives..., ALL_zeros (stored+implicit)..., positives...]
 *   |<- neg_end ->|<------- total_zero -------->|
 *
 * Rank offsets:
 *   negative at stored pos i : full pos = i              (no shift)
 *   positive at stored pos i : full pos = i + n_impl_zero (shift right)
 *   zeros                    : avg rank = neg_end + (total_zero+1)/2
 *
 * Shared-memory layout (doubles):
 *   grp_sums[n_groups]      rank-sum accumulators
 *   grp_nz_count[n_groups]  nonzero-per-group counters
 *   warp_buf[32]            tie-correction reduction scratch
 *
 * Grid: (sb_cols,)   Block: (tpb,)
 */
__global__ void rank_sums_sparse_ovr_kernel(
    const float* __restrict__ sorted_vals,
    const int* __restrict__ sorted_row_idx,
    const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, const double* __restrict__ group_sizes,
    double* __restrict__ rank_sums, double* __restrict__ tie_corr,
    double* __restrict__ nz_count_scratch, int n_rows, int sb_cols,
    int n_groups, bool compute_tie_corr, bool use_gmem) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];
    int nnz_stored = seg_end - seg_start;

    const float* sv = sorted_vals + seg_start;
    const int* si = sorted_row_idx + seg_start;

    extern __shared__ double smem[];
    double* grp_sums;
    double* grp_nz_count;
    // Accumulator stride: 1 for shared mem (dense per-block), sb_cols for
    // gmem (row-major layout (n_groups, sb_cols) shared across blocks).
    int acc_stride;

    if (use_gmem) {
        // Output rank_sums doubles as accumulator (pre-zeroed by caller).
        grp_sums = rank_sums + (size_t)col;
        grp_nz_count = nz_count_scratch + (size_t)col;
        acc_stride = sb_cols;
    } else {
        grp_sums = smem;
        grp_nz_count = smem + n_groups;
        acc_stride = 1;
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            grp_sums[g] = 0.0;
            grp_nz_count[g] = 0.0;
        }
        __syncthreads();
    }

    // --- Find zero range: neg_end = first val >= 0, pos_start = first val > 0
    // ---
    __shared__ int sh_neg_end;
    __shared__ int sh_pos_start;
    if (threadIdx.x == 0) {
        // Binary search: first index where sv[i] >= 0.0
        int lo = 0, hi = nnz_stored;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sv[mid] < 0.0f)
                lo = mid + 1;
            else
                hi = mid;
        }
        sh_neg_end = lo;
        // Binary search: first index where sv[i] > 0.0
        hi = nnz_stored;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sv[mid] <= 0.0f)
                lo = mid + 1;
            else
                hi = mid;
        }
        sh_pos_start = lo;
    }
    __syncthreads();

    int neg_end = sh_neg_end;
    int pos_start = sh_pos_start;
    int n_stored_zero = pos_start - neg_end;
    int n_implicit_zero = n_rows - nnz_stored;
    int total_zero = n_implicit_zero + n_stored_zero;
    double zero_avg_rank =
        (total_zero > 0) ? (double)neg_end + (total_zero + 1.0) / 2.0 : 0.0;

    // Rank offset for positive stored values:
    //   full_pos(i) = i + n_implicit_zero  for i >= pos_start
    // So avg_rank for tie group [a,b) of positives:
    //   = n_implicit_zero + (a + b + 1) / 2
    int offset_pos = n_implicit_zero;

    // --- Count stored values != 0.0 per group ---
    for (int i = threadIdx.x; i < nnz_stored; i += blockDim.x) {
        if (i < neg_end || i >= pos_start) {  // skip stored zeros
            int grp = group_codes[si[i]];
            if (grp < n_groups) {
                atomicAdd(&grp_nz_count[grp * acc_stride], 1.0);
            }
        }
    }
    __syncthreads();

    // --- Zero-rank contribution per group ---
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        double n_zero_in_g = group_sizes[g] - grp_nz_count[g * acc_stride];
        grp_sums[g * acc_stride] = n_zero_in_g * zero_avg_rank;
    }
    __syncthreads();

    // --- Walk ALL stored values, skip stored zeros, compute ranks ---
    // Chunk over [0, nnz_stored), skip [neg_end, pos_start).
    int chunk = (nnz_stored + blockDim.x - 1) / blockDim.x;
    int my_start = threadIdx.x * chunk;
    int my_end = my_start + chunk;
    if (my_end > nnz_stored) my_end = nnz_stored;

    double local_tie_sum = 0.0;

    int i = my_start;
    while (i < my_end) {
        // Skip stored zeros
        if (i >= neg_end && i < pos_start) {
            i = pos_start;
            continue;
        }

        float val = sv[i];

        int tie_local_end = i + 1;
        while (tie_local_end < my_end && sv[tie_local_end] == val)
            ++tie_local_end;
        // Don't let local tie range cross into stored-zero region
        if (val < 0.0f && tie_local_end > neg_end) tie_local_end = neg_end;

        int tie_global_start = i;
        if (i == my_start && i > 0 && sv[i - 1] == val) {
            // Binary search for first occurrence
            int search_lo = (val < 0.0f) ? 0 : pos_start;
            int lo = search_lo, hi = i;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (sv[mid] < val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            tie_global_start = lo;
        }
        // Handle thread resuming at pos_start after skipping zeros
        if (i == pos_start && i > 0 && pos_start > neg_end &&
            val == sv[pos_start] && i != my_start) {
            // Already at pos_start boundary, tie_global_start = pos_start
            tie_global_start = pos_start;
        }

        int tie_global_end = tie_local_end;
        if (tie_local_end == my_end && tie_local_end < nnz_stored &&
            tie_local_end != neg_end && sv[tie_local_end] == val) {
            int search_hi = (val < 0.0f) ? (neg_end - 1) : (nnz_stored - 1);
            int lo = tie_local_end, hi = search_hi;
            while (lo < hi) {
                int mid = hi - ((hi - lo) >> 1);
                if (sv[mid] > val)
                    hi = mid - 1;
                else
                    lo = mid;
            }
            tie_global_end = lo + 1;
        }

        int total_tie = tie_global_end - tie_global_start;

        // Rank depends on sign:
        //   negative (i < neg_end): full pos = stored pos (no shift)
        //   positive (i >= pos_start): full pos = stored pos + n_implicit_zero
        double avg_rank;
        if (val < 0.0f) {
            avg_rank = (double)(tie_global_start + tie_global_end + 1) / 2.0;
        } else {
            avg_rank = (double)offset_pos +
                       (double)(tie_global_start + tie_global_end + 1) / 2.0;
        }

        for (int j = i; j < tie_local_end; ++j) {
            int grp = group_codes[si[j]];
            if (grp < n_groups) {
                atomicAdd(&grp_sums[grp * acc_stride], avg_rank);
            }
        }

        if (compute_tie_corr && tie_global_start >= my_start && total_tie > 1) {
            double t = (double)total_tie;
            local_tie_sum += t * t * t - t;
        }

        i = tie_local_end;
    }

    __syncthreads();

    // Write rank sums to global output (smem path only — gmem path is direct)
    if (!use_gmem) {
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            rank_sums[(size_t)g * sb_cols + col] = grp_sums[g];
        }
    }

    // Tie correction: warp + block reduction
    if (compute_tie_corr) {
        // Zero tie group contribution (one thread only)
        if (threadIdx.x == 0 && total_zero > 1) {
            double tz = (double)total_zero;
            local_tie_sum += tz * tz * tz - tz;
        }

        // smem path: warp buf after both accumulator arrays (2 * n_groups).
        // gmem path: accumulators are in gmem, warp buf starts at smem[0].
        int warp_buf_off = use_gmem ? 0 : 2 * n_groups;
        double* warp_buf = smem + warp_buf_off;

#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local_tie_sum += __shfl_down_sync(0xffffffff, local_tie_sum, off);
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        if (lane == 0) warp_buf[wid] = local_tie_sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            double v = (threadIdx.x < ((blockDim.x + 31) >> 5))
                           ? warp_buf[threadIdx.x]
                           : 0.0;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) {
                double n = (double)n_rows;
                double denom = n * n * n - n;
                tie_corr[col] = (denom > 0.0) ? (1.0 - v / denom) : 1.0;
            }
        }
    }
}

/**
 * Pre-sort cast-and-accumulate kernel for dense OVR host streaming.
 *
 * Reads a sub-batch block in its native host dtype (InT = float or double),
 * writes a float32 copy used as the sort input, and accumulates per-group
 * sum, sum-of-squares and nonzero counts in float64.  Stats are derived
 * from the original-precision values so float64 host input keeps its
 * precision while the sort still runs on float32 keys.
 *
 * Block-per-column layout (grid: (sb_cols,), block: (tpb,)).
 * Shared memory: 3 * n_groups doubles (s_sum, s_sq, s_nnz).
 */
template <typename InT>
__global__ void ovr_cast_and_accumulate_dense_kernel(
    const InT* __restrict__ block_in, float* __restrict__ block_f32_out,
    const int* __restrict__ group_codes, double* __restrict__ group_sums,
    double* __restrict__ group_sq_sums, double* __restrict__ group_nnz,
    int n_rows, int sb_cols, int n_groups) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    extern __shared__ double smem[];
    double* s_sum = smem;
    double* s_sq = smem + n_groups;
    double* s_nnz = smem + 2 * n_groups;

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        s_sum[g] = 0.0;
        s_sq[g] = 0.0;
        s_nnz[g] = 0.0;
    }
    __syncthreads();

    const InT* src = block_in + (size_t)col * n_rows;
    float* dst = block_f32_out + (size_t)col * n_rows;

    for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
        InT v_in = src[r];
        double v = (double)v_in;
        dst[r] = (float)v_in;
        int g = group_codes[r];
        if (g < n_groups) {
            atomicAdd(&s_sum[g], v);
            atomicAdd(&s_sq[g], v * v);
            if (v != 0.0) atomicAdd(&s_nnz[g], 1.0);
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        group_sums[(size_t)g * sb_cols + col] = s_sum[g];
        group_sq_sums[(size_t)g * sb_cols + col] = s_sq[g];
        group_nnz[(size_t)g * sb_cols + col] = s_nnz[g];
    }
}

/**
 * One-shot cast-and-accumulate kernel for CSR-layout host streaming.
 *
 * The OVO CSR host path uploads the full CSR once; this kernel walks the
 * uploaded data row-by-row, writes a float32 copy of the values, and
 * accumulates per-group sum/sum-sq/nnz directly into a full-size
 * (n_groups_stats, n_cols) output using global atomics.  stats_codes[row]
 * must be in [0, n_groups_stats) to contribute; other values (e.g. the
 * sentinel for unselected cells) are skipped.
 *
 * Grid: (ceil(n_rows/tpb),), Block: (tpb,).
 */
template <typename InT>
__global__ void cast_and_accumulate_csr_kernel(
    const InT* __restrict__ data_in, float* __restrict__ data_f32_out,
    const int* __restrict__ indices, const int* __restrict__ indptr,
    const int* __restrict__ stats_codes, double* __restrict__ group_sums,
    double* __restrict__ group_sq_sums, double* __restrict__ group_nnz,
    int n_rows, int n_cols, int n_groups_stats) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int slot = stats_codes[row];
    int rs = indptr[row];
    int re = indptr[row + 1];
    bool accumulate = (slot >= 0 && slot < n_groups_stats);
    for (int p = rs; p < re; p++) {
        InT v_in = data_in[p];
        double v = (double)v_in;
        data_f32_out[p] = (float)v_in;
        if (accumulate) {
            int c = indices[p];
            atomicAdd(&group_sums[(size_t)slot * n_cols + c], v);
            atomicAdd(&group_sq_sums[(size_t)slot * n_cols + c], v * v);
            if (v != 0.0) {
                atomicAdd(&group_nnz[(size_t)slot * n_cols + c], 1.0);
            }
        }
    }
}

/**
 * Pre-sort cast-and-accumulate kernel for sparse OVR host streaming.
 *
 * Sub-batch CSC data is laid out contiguously: values for column c live
 * at positions [col_seg_offsets[c], col_seg_offsets[c+1]).  For each
 * stored value, read the native-dtype InT, write a float32 copy for the
 * CUB sort, and accumulate per-group sum/sum-sq/nnz in float64.  Implicit
 * zeros contribute nothing to any of these stats.
 *
 * Block-per-column layout (grid: (sb_cols,), block: (tpb,)).
 * Shared memory: 3 * n_groups doubles.
 */
template <typename InT>
__global__ void ovr_cast_and_accumulate_sparse_kernel(
    const InT* __restrict__ data_in, float* __restrict__ data_f32_out,
    const int* __restrict__ indices, const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, double* __restrict__ group_sums,
    double* __restrict__ group_sq_sums, double* __restrict__ group_nnz,
    int sb_cols, int n_groups) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];

    extern __shared__ double smem[];
    double* s_sum = smem;
    double* s_sq = smem + n_groups;
    double* s_nnz = smem + 2 * n_groups;

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        s_sum[g] = 0.0;
        s_sq[g] = 0.0;
        s_nnz[g] = 0.0;
    }
    __syncthreads();

    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) {
        InT v_in = data_in[i];
        double v = (double)v_in;
        data_f32_out[i] = (float)v_in;
        int row = indices[i];
        int g = group_codes[row];
        if (g < n_groups) {
            atomicAdd(&s_sum[g], v);
            atomicAdd(&s_sq[g], v * v);
            if (v != 0.0) atomicAdd(&s_nnz[g], 1.0);
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        group_sums[(size_t)g * sb_cols + col] = s_sum[g];
        group_sq_sums[(size_t)g * sb_cols + col] = s_sq[g];
        group_nnz[(size_t)g * sb_cols + col] = s_nnz[g];
    }
}
