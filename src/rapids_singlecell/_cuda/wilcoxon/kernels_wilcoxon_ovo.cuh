#pragma once

#include <cuda_runtime.h>

// ============================================================================
// CSR → dense F-order extraction
// ============================================================================

__global__ void csr_extract_dense_kernel(const double* __restrict__ data,
                                         const int* __restrict__ indices,
                                         const int* __restrict__ indptr,
                                         const int* __restrict__ row_ids,
                                         double* __restrict__ out, int n_target,
                                         int col_start, int col_stop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_target) return;

    int row = row_ids[tid];
    int rs = indptr[row];
    int re = indptr[row + 1];

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
        out[(long long)(c - col_start) * n_target + tid] = data[p];
    }
}

// ============================================================================
// Batched rank sums — pre-sorted (binary search, no shared memory sort)
// Used by the OVO streaming pipeline in wilcoxon_streaming.cu.
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

    double local_sum = 0.0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        double v = grp_col[i];
        int lo, hi;

        lo = 0;
        hi = n_ref;
        while (lo < hi) {
            int m = (lo + hi) >> 1;
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        lo = n_lt_ref;
        hi = n_ref;
        while (lo < hi) {
            int m = (lo + hi) >> 1;
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;
        lo = 0;
        hi = n_grp;
        while (lo < hi) {
            int m = (lo + hi) >> 1;
            if (grp_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_grp = lo;
        lo = n_lt_grp;
        hi = n_grp;
        while (lo < hi) {
            int m = (lo + hi) >> 1;
            if (grp_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    __shared__ double warp_buf[32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = local_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        double val = (threadIdx.x < ((blockDim.x + 31) >> 5))
                         ? warp_buf[threadIdx.x]
                         : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = val;
    }

    if (!compute_tie_corr) return;
    __syncthreads();

    if (threadIdx.x == 0) {
        int ri = 0, gi = 0;
        double tie_sum = 0.0;
        while (ri < n_ref || gi < n_grp) {
            double v;
            if (gi >= n_grp)
                v = ref_col[ri];
            else if (ri >= n_ref)
                v = grp_col[gi];
            else
                v = (ref_col[ri] <= grp_col[gi]) ? ref_col[ri] : grp_col[gi];
            int cnt = 0;
            while (ri < n_ref && ref_col[ri] == v) {
                ++ri;
                ++cnt;
            }
            while (gi < n_grp && grp_col[gi] == v) {
                ++gi;
                ++cnt;
            }
            if (cnt > 1) {
                double t = (double)cnt;
                tie_sum += t * t * t - t;
            }
        }
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        tie_corr[grp * n_cols + col] =
            (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

// ============================================================================
// Grouped statistics: sum, sum-of-squares, nnz per group
// ============================================================================

__global__ void grouped_stats_kernel(
    const double* __restrict__ data,      // F-order (n_all_rows, n_cols)
    const int* __restrict__ grp_offsets,  // (n_groups + 1,)
    double* __restrict__ sums,            // (n_groups, n_cols) row-major
    double* __restrict__ sq_sums,         // (n_groups, n_cols) row-major
    double* __restrict__ nnz_counts,      // (n_groups, n_cols) row-major
    int n_all_rows, int n_cols, int n_groups, bool compute_nnz) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

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

    const double* col_data = data + (long long)col * n_all_rows;

    for (int g = 0; g < n_groups; g++) {
        int g_start = grp_offsets[g];
        int g_end = grp_offsets[g + 1];

        double local_sum = 0.0;
        double local_sq = 0.0;
        double local_nnz = 0.0;

        for (int i = g_start + threadIdx.x; i < g_end; i += blockDim.x) {
            double v = col_data[i];
            local_sum += v;
            local_sq += v * v;
            if (compute_nnz && v != 0.0) local_nnz += 1.0;
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
            local_sq += __shfl_down_sync(0xffffffff, local_sq, off);
            if (compute_nnz)
                local_nnz += __shfl_down_sync(0xffffffff, local_nnz, off);
        }

        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&s_sum[g], local_sum);
            atomicAdd(&s_sq[g], local_sq);
            if (compute_nnz) atomicAdd(&s_nnz[g], local_nnz);
        }
        __syncthreads();
    }

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        sums[(long long)g * n_cols + col] = s_sum[g];
        sq_sums[(long long)g * n_cols + col] = s_sq[g];
        if (compute_nnz) nnz_counts[(long long)g * n_cols + col] = s_nnz[g];
    }
}
