#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ double wilcoxon_block_sum(double val,
                                                     double* warp_buf) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v = (threadIdx.x < ((blockDim.x + 31) >> 5))
                       ? warp_buf[threadIdx.x]
                       : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        return v;
    }
    return 0.0;
}

/**
 * Kernel to compute tie correction factor for Wilcoxon test.
 * Formula: tc = 1 - sum(t^3 - t) / (n^3 - n) where t is the count of tied
 * values.
 *
 * Each block handles one column. Uses binary search to find tie groups.
 * Assumes input is sorted column-wise (F-order).
 */
__global__ void tie_correction_kernel(const double* __restrict__ sorted_vals,
                                      double* __restrict__ correction,
                                      const int n_rows, const int n_cols) {
    // Each block handles one column
    int col = blockIdx.x;
    if (col >= n_cols) return;

    const double* sv = sorted_vals + (size_t)col * n_rows;

    double local_sum = 0.0;
    int tid = threadIdx.x;

    // Each thread processes positions where it detects END of a tie group
    // Start from index 1, check if sv[i-1] != sv[i] (boundary detected)
    // When at boundary, use binary search to find tie group size
    for (int i = tid + 1; i <= n_rows; i += blockDim.x) {
        // Detect boundary: either at the end, or value changed
        bool at_boundary = (i == n_rows) || (sv[i] != sv[i - 1]);

        if (at_boundary) {
            // Found end of tie group at position i-1
            // Binary search for start of this tie group
            double val = sv[i - 1];
            int lo = 0, hi = i - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (sv[mid] < val) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            int tie_count = i - lo;

            // t^3 - t for this tie group
            double t = (double)tie_count;
            local_sum += t * t * t - t;
        }
    }

    // Warp-level reduction using shuffle
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Cross-warp reduction using small shared memory
    __shared__ double warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction in first warp
    // Note: blockDim.x must be a multiple of 32 for correct warp reduction
    if (tid < 32) {
        double val = (tid < (blockDim.x >> 5)) ? warp_sums[tid] : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            double n = (double)n_rows;
            double denom = n * n * n - n;
            if (denom > 0) {
                correction[col] = 1.0 - val / denom;
            } else {
                correction[col] = 1.0;
            }
        }
    }
}

/**
 * Kernel to compute average ranks for each column.
 * Uses scipy.stats.rankdata 'average' method: ties get the average of the ranks
 * they would span.
 *
 * Each block handles one column. Assumes input is sorted column-wise (F-order).
 */
__global__ void average_rank_kernel(const double* __restrict__ sorted_vals,
                                    const int* __restrict__ sorter,
                                    double* __restrict__ ranks,
                                    const int n_rows, const int n_cols) {
    // Each thread block handles one column
    int col = blockIdx.x;
    if (col >= n_cols) return;

    // Pointers to this column's data
    const double* sv = sorted_vals + (size_t)col * n_rows;
    const int* si = sorter + (size_t)col * n_rows;
    double* rk = ranks + (size_t)col * n_rows;

    // Each thread processes multiple rows
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        double val = sv[i];

        // Binary search for tie_start (first element equal to val)
        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (sv[mid] < val) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int tie_start = lo;

        // Binary search for tie_end (last element equal to val)
        lo = i;
        hi = n_rows - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (sv[mid] > val) {
                hi = mid - 1;
            } else {
                lo = mid;
            }
        }
        int tie_end = lo;

        // Average rank for ties: (start + end + 2) / 2 (1-based ranks)
        double avg_rank = (double)(tie_start + tie_end + 2) / 2.0;

        // Write rank to original position
        rk[si[i]] = avg_rank;
    }
}

/**
 * OVO dense rank core.
 *
 * ref_sorted is F-order and sorted independently for every column.
 * grp_data is F-order and contains test-group rows concatenated by
 * grp_offsets.  One block computes one (column, test-group) result.
 *
 * This intentionally centralizes the OVO math; host/device and CSR/CSC/dense
 * paths only need to materialize bounded dense column batches that feed this
 * kernel.
 */
__global__ void ovo_rank_dense_kernel(const float* __restrict__ ref_sorted,
                                      const float* __restrict__ grp_data,
                                      const int* __restrict__ grp_offsets,
                                      double* __restrict__ rank_sums,
                                      double* __restrict__ tie_corr, int n_ref,
                                      int n_all_grp, int n_cols, int n_groups,
                                      bool compute_tie_corr) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_data + (long long)col * n_all_grp + g_start;

    __shared__ double warp_buf[32];
    double local_rank = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];

        int lo = 0, hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;

        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;

        int n_lt_grp = 0;
        int n_eq_grp = 0;
        for (int j = 0; j < n_grp; ++j) {
            float u = grp_col[j];
            n_lt_grp += (u < v);
            n_eq_grp += (u == v);
        }

        local_rank += (double)(n_lt_ref + n_lt_grp) +
                      ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    double total_rank = wilcoxon_block_sum(local_rank, warp_buf);
    if (threadIdx.x == 0) {
        rank_sums[(size_t)grp * n_cols + col] = total_rank;
    }

    if (!compute_tie_corr) return;
    __syncthreads();

    double local_tie = 0.0;

    for (int i = threadIdx.x; i < n_ref; i += blockDim.x) {
        if (i == 0 || ref_col[i] != ref_col[i - 1]) {
            float v = ref_col[i];
            int lo = i + 1, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int count = lo - i;
            for (int j = 0; j < n_grp; ++j) count += (grp_col[j] == v);
            if (count > 1) {
                double t = (double)count;
                local_tie += t * t * t - t;
            }
        }
    }

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        bool seen_in_group = false;
        for (int j = 0; j < i; ++j) {
            if (grp_col[j] == v) {
                seen_in_group = true;
                break;
            }
        }
        if (seen_in_group) continue;

        int lo = 0, hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        if (lo < n_ref && ref_col[lo] == v) continue;

        int count = 0;
        for (int j = 0; j < n_grp; ++j) count += (grp_col[j] == v);
        if (count > 1) {
            double t = (double)count;
            local_tie += t * t * t - t;
        }
    }

    double tie_sum = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        tie_corr[(size_t)grp * n_cols + col] =
            (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

__global__ void ovo_rank_presorted_kernel(const float* __restrict__ ref_sorted,
                                          const float* __restrict__ grp_sorted,
                                          const int* __restrict__ grp_offsets,
                                          double* __restrict__ rank_sums,
                                          double* __restrict__ tie_corr,
                                          int n_ref, int n_all_grp, int n_cols,
                                          int n_groups, bool compute_tie_corr) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_sorted + (long long)col * n_all_grp + g_start;

    __shared__ double warp_buf[32];
    double local_rank = 0.0;

    int ref_lb = 0, ref_ub = 0;
    int grp_lb = 0, grp_ub = 0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];

        int lo = ref_lb, hi = n_ref;
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
            if (grp_col[m] < v)
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
            if (grp_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;
        grp_ub = lo;

        local_rank += (double)(n_lt_ref + n_lt_grp) +
                      ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    double total_rank = wilcoxon_block_sum(local_rank, warp_buf);
    if (threadIdx.x == 0) {
        rank_sums[(size_t)grp * n_cols + col] = total_rank;
    }

    if (!compute_tie_corr) return;
    __syncthreads();

    double local_tie = 0.0;
    int grp_lb_tie = 0, grp_ub_tie = 0;
    for (int i = threadIdx.x; i < n_ref; i += blockDim.x) {
        if (i == 0 || ref_col[i] != ref_col[i - 1]) {
            float v = ref_col[i];
            int lo = i + 1, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_ref = lo - i;

            lo = grp_lb_tie;
            hi = n_grp;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (grp_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int lb = lo;
            grp_lb_tie = lb;

            lo = (grp_ub_tie > lb) ? grp_ub_tie : lb;
            hi = n_grp;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (grp_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_grp = lo - lb;
            grp_ub_tie = lo;

            int cnt = cnt_ref + cnt_grp;
            if (cnt > 1) {
                double t = (double)cnt;
                local_tie += t * t * t - t;
            }
        }
    }

    int ref_lb_tie = 0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        if (i == 0 || grp_col[i] != grp_col[i - 1]) {
            float v = grp_col[i];
            int lo = ref_lb_tie, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            ref_lb_tie = lo;
            if (lo < n_ref && ref_col[lo] == v) continue;

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

    double tie_sum = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        tie_corr[(size_t)grp * n_cols + col] =
            (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

/**
 * OVR dense rank core.
 *
 * sorted_vals and sorter are F-order outputs of sorting each column of the
 * current dense block.  The kernel directly accumulates rank sums per group,
 * avoiding a full ranks matrix and a group one-hot matrix multiply.
 */
__global__ void ovr_rank_dense_kernel(const float* __restrict__ sorted_vals,
                                      const int* __restrict__ sorter,
                                      const int* __restrict__ group_codes,
                                      double* __restrict__ rank_sums,
                                      double* __restrict__ tie_corr, int n_rows,
                                      int n_cols, int n_groups,
                                      bool compute_tie_corr) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    const float* sv = sorted_vals + (long long)col * n_rows;
    const int* si = sorter + (long long)col * n_rows;

    double local_tie = 0.0;
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        float val = sv[i];

        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sv[mid] < val)
                lo = mid + 1;
            else
                hi = mid;
        }
        int tie_start = lo;

        lo = i;
        hi = n_rows - 1;
        while (lo < hi) {
            int mid = lo + ((hi - lo + 1) >> 1);
            if (sv[mid] > val)
                hi = mid - 1;
            else
                lo = mid;
        }
        int tie_end = lo;
        double avg_rank = (double)(tie_start + tie_end + 2) / 2.0;

        int row = si[i];
        int group = group_codes[row];
        if (group >= 0 && group < n_groups) {
            atomicAdd(&rank_sums[(size_t)group * n_cols + col], avg_rank);
        }

        if (compute_tie_corr && i == tie_end) {
            double t = (double)(tie_end - tie_start + 1);
            if (t > 1.0) local_tie += t * t * t - t;
        }
    }

    if (!compute_tie_corr) return;

    __shared__ double warp_buf[32];
    double tie_sum = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        double n = (double)n_rows;
        double denom = n * n * n - n;
        tie_corr[col] = (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}
