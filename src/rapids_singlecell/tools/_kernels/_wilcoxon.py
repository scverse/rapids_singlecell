from __future__ import annotations

import cupy as cp

_tie_correction_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void tie_correction_kernel(
    const double* __restrict__ sorted_vals,
    double* __restrict__ correction,
    const int n_rows,
    const int n_cols)
{
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
""",
    "tie_correction_kernel",
)


_rank_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void average_rank_kernel(
    const double* __restrict__ sorted_vals,
    const long long* __restrict__ sorter,
    double* __restrict__ ranks,
    const int n_rows,
    const int n_cols)
{
    // Each thread block handles one column
    int col = blockIdx.x;
    if (col >= n_cols) return;

    // Pointers to this column's data
    const double* sv = sorted_vals + (size_t)col * n_rows;
    const long long* si = sorter + (size_t)col * n_rows;
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
""",
    "average_rank_kernel",
)
