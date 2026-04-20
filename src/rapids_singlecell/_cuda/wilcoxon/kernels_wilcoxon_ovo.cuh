#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Warp reduction helper (sum doubles across block via warp_buf)
// ============================================================================

__device__ __forceinline__ double block_reduce_sum(double val,
                                                   double* warp_buf) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v2 = (threadIdx.x < ((blockDim.x + 31) >> 5))
                        ? warp_buf[threadIdx.x]
                        : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v2 += __shfl_down_sync(0xffffffff, v2, off);
        return v2;  // only lane 0 of warp 0 has the final result
    }
    return 0.0;
}

// ============================================================================
// Parallel tie correction — all threads collaborate.
//
// For each unique value in the combined sorted (ref, grp) arrays, accumulate
// t^3 - t where t = count of that value.  Uses two passes:
//   1. Iterate unique values in ref_col, count in both arrays.
//   2. Iterate unique values in grp_col that do NOT appear in ref_col.
//
// Incremental binary search bounds exploit monotonicity within each thread's
// stride to reduce total search work.
//
// Caller must __syncthreads() before calling.  warp_buf is reused for
// reduction (32 doubles, shared memory).
// ============================================================================

__device__ __forceinline__ void compute_tie_correction_parallel(
    const float* ref_col, int n_ref, const float* grp_col, int n_grp,
    double* warp_buf, double* out) {
    double local_tie = 0.0;

    // Pass 1: unique values in ref_col
    int grp_lb = 0, grp_ub = 0;
    for (int i = threadIdx.x; i < n_ref; i += blockDim.x) {
        if (i == 0 || ref_col[i] != ref_col[i - 1]) {
            float v = ref_col[i];

            // Count in ref: upper_bound from i+1
            int lo = i + 1, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_ref = lo - i;

            // Count in grp: incremental lower/upper bound
            lo = grp_lb;
            hi = n_grp;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (grp_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int lb = lo;
            grp_lb = lb;

            lo = (grp_ub > lb) ? grp_ub : lb;
            hi = n_grp;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (grp_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_grp = lo - lb;
            grp_ub = lo;

            int cnt = cnt_ref + cnt_grp;
            if (cnt > 1) {
                double t = (double)cnt;
                local_tie += t * t * t - t;
            }
        }
    }

    // Pass 2: unique values in grp_col that are absent from ref_col
    int ref_lb = 0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        if (i == 0 || grp_col[i] != grp_col[i - 1]) {
            float v = grp_col[i];

            // Incremental lower_bound in ref
            int lo = ref_lb, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            ref_lb = lo;

            if (lo >= n_ref || ref_col[lo] != v) {
                // Value not in ref — count in grp only (upper_bound from i+1)
                lo = i + 1;
                hi = n_grp;
                while (lo < hi) {
                    int m = lo + ((hi - lo) >> 1);
                    if (grp_col[m] <= v)
                        lo = m + 1;
                    else
                        hi = m;
                }
                int cnt = lo - i;
                if (cnt > 1) {
                    double t = (double)cnt;
                    local_tie += t * t * t - t;
                }
            }
        }
    }

    // Block-wide reduction
    double tie_sum = block_reduce_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        *out = (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

// ============================================================================
// Batched rank sums — pre-sorted (binary search, no shared memory sort)
// Used by the OVO streaming pipeline in wilcoxon_streaming.cu.
//
// Incremental binary search: each thread carries forward lower/upper bound
// positions across loop iterations, exploiting the monotonicity of the
// sorted grp_col values within each thread's stride.
// ============================================================================

__global__ void batched_rank_sums_presorted_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_sorted,
    const int* __restrict__ grp_offsets, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    if (n_grp == 0) {
        if (threadIdx.x == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_sorted + (long long)col * n_all_grp + g_start;

    // Incremental binary search bounds (advance monotonically per thread)
    int ref_lb = 0, ref_ub = 0;
    int grp_lb = 0, grp_ub = 0;
    double local_sum = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        int lo, hi;

        // Lower bound in ref (from ref_lb)
        lo = ref_lb;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        ref_lb = n_lt_ref;

        // Upper bound in ref (from max(ref_ub, n_lt_ref))
        lo = (ref_ub > n_lt_ref) ? ref_ub : n_lt_ref;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;
        ref_ub = lo;

        // Lower bound in grp (from grp_lb)
        lo = grp_lb;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_grp = lo;
        grp_lb = n_lt_grp;

        // Upper bound in grp (from max(grp_ub, n_lt_grp))
        lo = (grp_ub > n_lt_grp) ? grp_ub : n_lt_grp;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;
        grp_ub = lo;

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    __shared__ double warp_buf[32];
    double total = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    compute_tie_correction_parallel(ref_col, n_ref, grp_col, n_grp, warp_buf,
                                    &tie_corr[grp * n_cols + col]);
}

// ============================================================================
// Tier 1 fused kernel: smem bitonic sort + binary search rank sums
// For small groups (< ~2K cells).  No CUB, no global memory sort buffers.
// Grid: (n_cols, n_groups), Block: min(padded_grp_size, 512)
// Shared memory: padded_grp_size floats + 32 doubles (warp reduction)
// ============================================================================

__global__ void ovo_fused_sort_rank_kernel(
    const float* __restrict__ ref_sorted,  // F-order (n_ref, n_cols) sorted
    const float* __restrict__ grp_dense,   // F-order (n_all_grp, n_cols)
                                           // unsorted
    const int* __restrict__ grp_offsets,   // (n_groups + 1,)
    double* __restrict__ rank_sums,        // (n_groups, n_cols) row-major
    double* __restrict__ tie_corr,         // (n_groups, n_cols) row-major
    int n_ref, int n_all_grp, int n_cols, int n_groups, bool compute_tie_corr,
    int padded_grp_size) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    if (n_grp == 0) {
        if (threadIdx.x == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    // Shared memory: [padded_grp_size floats | 32 doubles for warp reduction]
    extern __shared__ char smem_raw[];
    float* grp_smem = (float*)smem_raw;
    double* warp_buf = (double*)(smem_raw + padded_grp_size * sizeof(float));

    // Load group data into shared memory, pad with +INF
    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x)
        grp_smem[i] = grp_col[i];
    for (int i = n_grp + threadIdx.x; i < padded_grp_size; i += blockDim.x)
        grp_smem[i] = __int_as_float(0x7f800000);  // +INF
    __syncthreads();

    // Bitonic sort in shared memory
    for (int k = 2; k <= padded_grp_size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < padded_grp_size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool asc = ((i & k) == 0);
                    float a = grp_smem[i], b = grp_smem[ixj];
                    if (asc ? (a > b) : (a < b)) {
                        grp_smem[i] = b;
                        grp_smem[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Binary search each sorted grp element against sorted ref
    // Incremental bounds: values are monotonic within each thread's stride
    const float* ref_col = ref_sorted + (long long)col * n_ref;
    int ref_lb = 0, ref_ub = 0;
    int grp_lb = 0, grp_ub = 0;
    double local_sum = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_smem[i];
        int lo, hi;

        lo = ref_lb;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        ref_lb = n_lt_ref;

        lo = (ref_ub > n_lt_ref) ? ref_ub : n_lt_ref;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;
        ref_ub = lo;

        lo = grp_lb;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_smem[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_grp = lo;
        grp_lb = n_lt_grp;

        lo = (grp_ub > n_lt_grp) ? grp_ub : n_lt_grp;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_smem[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;
        grp_ub = lo;

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    // Block reduction → write rank_sums
    double total = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    // Parallel tie correction (grp_smem is sorted shared memory)
    compute_tie_correction_parallel(ref_col, n_ref, grp_smem, n_grp, warp_buf,
                                    &tie_corr[grp * n_cols + col]);
}
