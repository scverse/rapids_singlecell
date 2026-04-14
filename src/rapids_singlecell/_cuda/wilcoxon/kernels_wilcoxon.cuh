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
 * When use_gmem is false (default), per-group accumulators live in shared
 * memory (fast atomics, limited to ~750 groups on 48 KB devices).
 * When use_gmem is true, accumulators write directly to the output arrays
 * in global memory, supporting an arbitrary number of groups.  The caller
 * must pre-zero rank_sums (and group_sums/group_sq_sums/group_nnz if
 * compute_stats) before launching.
 *
 * Shared memory layout:
 *   use_gmem=false: (4 * n_groups + 32) doubles   (accumulators + warp buf)
 *   use_gmem=true:  32 doubles                     (warp buf only)
 */
__global__ void rank_sums_from_sorted_kernel(
    const float* __restrict__ sorted_vals,
    const int* __restrict__ sorted_row_idx, const int* __restrict__ group_codes,
    double* __restrict__ rank_sums, double* __restrict__ tie_corr,
    double* __restrict__ group_sums, double* __restrict__ group_sq_sums,
    double* __restrict__ group_nnz, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, bool compute_stats, bool use_gmem) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    extern __shared__ double smem[];

    // Accumulator pointers: shared memory (fast) or global memory (large
    // groups)
    double* grp_sums;
    double* s_sum;
    double* s_sq;
    double* s_nnz;

    if (use_gmem) {
        // Global memory path: write directly to output arrays (must be
        // pre-zeroed)
        grp_sums = rank_sums + (size_t)col;  // stride: n_cols
        s_sum = group_sums ? group_sums + (size_t)col : nullptr;
        s_sq = group_sq_sums ? group_sq_sums + (size_t)col : nullptr;
        s_nnz = group_nnz ? group_nnz + (size_t)col : nullptr;
    } else {
        // Shared memory path: per-block accumulators
        grp_sums = smem;
        s_sum = smem + n_groups;
        s_sq = smem + 2 * n_groups;
        s_nnz = smem + 3 * n_groups;

        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            grp_sums[g] = 0.0;
            if (compute_stats) {
                s_sum[g] = 0.0;
                s_sq[g] = 0.0;
                s_nnz[g] = 0.0;
            }
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
                int mid = (lo + hi) / 2;
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
                int mid = (lo + hi + 1) / 2;
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
                if (compute_stats) {
                    double v = (double)sv[j];
                    atomicAdd(&s_sum[grp * acc_stride], v);
                    atomicAdd(&s_sq[grp * acc_stride], v * v);
                    if (v != 0.0) atomicAdd(&s_nnz[grp * acc_stride], 1.0);
                }
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
            if (compute_stats) {
                group_sums[(size_t)g * n_cols + col] = s_sum[g];
                group_sq_sums[(size_t)g * n_cols + col] = s_sq[g];
                group_nnz[(size_t)g * n_cols + col] = s_nnz[g];
            }
        }
    }

    if (compute_tie_corr) {
        // Warp buf sits after all accumulator arrays in shared memory.
        // gmem path: accumulators are in global mem, warp buf starts at
        // smem[0]. smem path: 4 arrays of n_groups doubles, then warp buf.
        int warp_buf_off = use_gmem ? 0 : (compute_stats ? 4 : 1) * n_groups;
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
