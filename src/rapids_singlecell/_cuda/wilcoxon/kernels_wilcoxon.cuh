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
 * Used by the OVR streaming pipeline in wilcoxon_streaming.cu.
 */
__global__ void rank_sums_from_sorted_kernel(
    const float* __restrict__ sorted_vals,   // F-order (n_rows, n_cols)
    const int* __restrict__ sorted_row_idx,  // F-order (n_rows, n_cols)
    const int* __restrict__ group_codes,     // (n_rows_total,)
    double* __restrict__ rank_sums,          // (n_groups, n_cols) row-major
    double* __restrict__ tie_corr,           // (n_cols,)
    double* __restrict__ group_sums,         // (n_groups, n_cols) or NULL
    double* __restrict__ group_sq_sums,      // (n_groups, n_cols) or NULL
    double* __restrict__ group_nnz,          // (n_groups, n_cols) or NULL
    int n_rows, int n_cols, int n_groups, bool compute_tie_corr,
    bool compute_stats) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    extern __shared__ double smem[];
    double* grp_sums = smem;
    double* s_sum = smem + n_groups;
    double* s_sq = smem + 2 * n_groups;
    double* s_nnz = smem + 3 * n_groups;

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        grp_sums[g] = 0.0;
        if (compute_stats) {
            s_sum[g] = 0.0;
            s_sq[g] = 0.0;
            s_nnz[g] = 0.0;
        }
    }
    __syncthreads();

    const float* sv = sorted_vals + (size_t)col * n_rows;
    const int* si = sorted_row_idx + (size_t)col * n_rows;

    int chunk = (n_rows + blockDim.x - 1) / blockDim.x;
    int my_start = threadIdx.x * chunk;
    int my_end = my_start + chunk;
    if (my_end > n_rows) my_end = n_rows;

    double local_tie_sum = 0.0;

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
                atomicAdd(&grp_sums[grp], avg_rank);
                if (compute_stats) {
                    double v = (double)sv[j];
                    atomicAdd(&s_sum[grp], v);
                    atomicAdd(&s_sq[grp], v * v);
                    if (v != 0.0) atomicAdd(&s_nnz[grp], 1.0);
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

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        rank_sums[(size_t)g * n_cols + col] = grp_sums[g];
        if (compute_stats) {
            group_sums[(size_t)g * n_cols + col] = s_sum[g];
            group_sq_sums[(size_t)g * n_cols + col] = s_sq[g];
            group_nnz[(size_t)g * n_cols + col] = s_nnz[g];
        }
    }

    if (compute_tie_corr) {
        double* warp_buf = smem + n_groups;
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
