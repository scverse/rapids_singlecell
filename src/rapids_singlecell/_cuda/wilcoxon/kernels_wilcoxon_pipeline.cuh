#pragma once

#include <cuda_runtime.h>

/**
 * Convert a CSC column slice to a dense F-order float64 matrix.
 *
 * Each block handles one column. Threads first zero the output, then scatter
 * non-zero values from the CSC arrays into the correct positions.
 *
 * Template parameter T is the CSC data type (float or double).
 */
template <typename T>
__global__ void csc_slice_to_dense_kernel(
    const T* __restrict__ csc_data, const int* __restrict__ csc_indices,
    const int* __restrict__ csc_indptr,  // already offset to col_start
    double* __restrict__ dense,          // F-order (n_rows, n_cols)
    const int n_rows, const int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    double* out_col = dense + (size_t)col * n_rows;

    // Zero the output column
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        out_col[i] = 0.0;
    }
    __syncthreads();

    // Scatter non-zeros
    int start = csc_indptr[col];
    int end = csc_indptr[col + 1];
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        int row = csc_indices[j];
        out_col[row] = static_cast<double>(csc_data[j]);
    }
}

/**
 * Compute rank sums per group for "vs rest" mode.
 *
 * Uses a CSR-like group mapping: cat_offsets[g] .. cat_offsets[g+1] are indices
 * into cell_indices[], which gives the cell (row) positions for group g.
 *
 * Grid: (n_genes, n_groups). Each block computes the rank sum for one
 * (group, gene) pair using warp reduction.
 */
__global__ void rank_sum_grouped_kernel(
    const double* __restrict__ ranks,  // F-order (n_cells, n_genes)
    const int* __restrict__ cell_indices, const int* __restrict__ cat_offsets,
    double* __restrict__ rank_sums,  // (n_groups, n_genes) row-major
    const int n_cells, const int n_genes, const int n_groups) {
    int gene = blockIdx.x;
    int group = blockIdx.y;
    if (gene >= n_genes || group >= n_groups) return;

    const double* ranks_col = ranks + (size_t)gene * n_cells;

    int g_start = cat_offsets[group];
    int g_end = cat_offsets[group + 1];

    double local_sum = 0.0;
    for (int i = g_start + threadIdx.x; i < g_end; i += blockDim.x) {
        int cell = cell_indices[i];
        local_sum += ranks_col[cell];
    }

    // Warp reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Cross-warp reduction
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < ((blockDim.x + 31) >> 5))
                         ? warp_sums[threadIdx.x]
                         : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            rank_sums[(size_t)group * n_genes + gene] = val;
        }
    }
}

/**
 * Fused z-score + p-value kernel for "vs rest" mode.
 *
 * One thread per (group, gene). Computes:
 *   expected = group_size * (n_cells + 1) / 2
 *   variance = tie_corr[gene] * group_size * rest_size * (n_cells + 1) / 12
 *   z = (rank_sum - expected [- continuity]) / sqrt(variance)
 *   p = erfc(|z| / sqrt(2))
 */
__global__ void zscore_pvalue_vs_rest_kernel(
    const double* __restrict__ rank_sums,    // (n_groups, n_genes) row-major
    const double* __restrict__ tie_corr,     // (n_genes,)
    const double* __restrict__ group_sizes,  // (n_groups,)
    double* __restrict__ z_out,              // (n_groups, n_genes) row-major
    double* __restrict__ p_out,              // (n_groups, n_genes) row-major
    const int n_cells, const int n_genes, const int n_groups,
    const bool use_continuity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_groups * n_genes;
    if (idx >= total) return;

    int group = idx / n_genes;
    int gene = idx % n_genes;

    double gs = group_sizes[group];
    double rs = (double)n_cells - gs;
    double n1 = (double)n_cells + 1.0;

    double expected = gs * n1 * 0.5;
    double variance = tie_corr[gene] * gs * rs * n1 / 12.0;
    double std_dev = sqrt(variance);

    double diff = rank_sums[idx] - expected;
    if (use_continuity) {
        double sign = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
        double abs_diff = fabs(diff) - 0.5;
        if (abs_diff < 0.0) abs_diff = 0.0;
        diff = sign * abs_diff;
    }

    double z = (std_dev > 0.0) ? (diff / std_dev) : 0.0;
    // nan_to_num: if variance is 0, z is already 0
    double p = erfc(fabs(z) * M_SQRT1_2);  // M_SQRT1_2 = 1/sqrt(2)

    z_out[idx] = z;
    p_out[idx] = p;
}

/**
 * Compute group statistics (sum, sum-of-squares, nnz) from a dense F-order
 * matrix.  Same grid structure as rank_sum_grouped_kernel:
 * grid (chunk_genes, n_groups), one block per (gene, group).
 *
 * Uses warp + cross-warp reduction for three quantities simultaneously.
 * Results are written into row-major accumulators at the given gene_offset
 * within an output of width out_stride.
 */
__global__ void stats_grouped_kernel(
    const double* __restrict__ dense,  // F-order (n_cells, chunk_genes)
    const int* __restrict__ cell_indices, const int* __restrict__ cat_offsets,
    double* __restrict__ sums_out,     // (n_groups, out_stride) row-major
    double* __restrict__ sq_sums_out,  // (n_groups, out_stride) row-major
    double* __restrict__ nnz_out,      // (n_groups, out_stride) row-major
    const int n_cells, const int chunk_genes, const int n_groups,
    const int gene_offset, const int out_stride) {
    int gene = blockIdx.x;
    int group = blockIdx.y;
    if (gene >= chunk_genes || group >= n_groups) return;

    const double* col = dense + (size_t)gene * n_cells;
    int g_start = cat_offsets[group];
    int g_end = cat_offsets[group + 1];

    double local_sum = 0.0;
    double local_sq = 0.0;
    double local_nnz = 0.0;

    for (int i = g_start + threadIdx.x; i < g_end; i += blockDim.x) {
        int cell = cell_indices[i];
        double val = col[cell];
        local_sum += val;
        local_sq += val * val;
        local_nnz += (val != 0.0) ? 1.0 : 0.0;
    }

    // Warp reduction for all three quantities
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
        local_nnz += __shfl_down_sync(0xffffffff, local_nnz, offset);
    }

    // Cross-warp reduction
    __shared__ double ws_sum[32], ws_sq[32], ws_nnz[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        ws_sum[warp_id] = local_sum;
        ws_sq[warp_id] = local_sq;
        ws_nnz[warp_id] = local_nnz;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int n_warps = (blockDim.x + 31) >> 5;
        double s = (threadIdx.x < n_warps) ? ws_sum[threadIdx.x] : 0.0;
        double q = (threadIdx.x < n_warps) ? ws_sq[threadIdx.x] : 0.0;
        double z = (threadIdx.x < n_warps) ? ws_nnz[threadIdx.x] : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, offset);
            q += __shfl_down_sync(0xffffffff, q, offset);
            z += __shfl_down_sync(0xffffffff, z, offset);
        }
        if (threadIdx.x == 0) {
            size_t out_idx = (size_t)group * out_stride + gene_offset + gene;
            sums_out[out_idx] = s;
            sq_sums_out[out_idx] = q;
            nnz_out[out_idx] = z;
        }
    }
}

/**
 * Convert a CSC column slice to a dense F-order matrix with row filtering.
 *
 * Like csc_slice_to_dense_kernel, but uses a row_map to filter rows.
 * row_map[old_row] = new_row index (0..n_filtered-1) or -1 to skip.
 * Output dense has n_filtered rows.
 */
template <typename T>
__global__ void csc_slice_to_dense_filtered_kernel(
    const T* __restrict__ csc_data, const int* __restrict__ csc_indices,
    const int* __restrict__ csc_indptr,  // already offset to col_start
    const int* __restrict__ row_map,  // (n_total_rows,) maps old → new or -1
    double* __restrict__ dense,       // F-order (n_filtered, n_cols)
    const int n_filtered, const int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    double* out_col = dense + (size_t)col * n_filtered;

    // Zero the output column
    for (int i = threadIdx.x; i < n_filtered; i += blockDim.x) {
        out_col[i] = 0.0;
    }
    __syncthreads();

    // Scatter non-zeros using row_map
    int start = csc_indptr[col];
    int end = csc_indptr[col + 1];
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        int old_row = csc_indices[j];
        int new_row = row_map[old_row];
        if (new_row >= 0) {
            out_col[new_row] = static_cast<double>(csc_data[j]);
        }
    }
}

/**
 * Compute rank sum for "with reference" mode using a boolean mask.
 *
 * One block per gene. Sums ranks where group_mask[cell] == true.
 */
__global__ void rank_sum_masked_kernel(
    const double* __restrict__ ranks,     // F-order (n_combined, n_genes)
    const bool* __restrict__ group_mask,  // (n_combined,)
    double* __restrict__ rank_sums,       // (n_genes,)
    const int n_combined, const int n_genes) {
    int gene = blockIdx.x;
    if (gene >= n_genes) return;

    const double* ranks_col = ranks + (size_t)gene * n_combined;

    double local_sum = 0.0;
    for (int i = threadIdx.x; i < n_combined; i += blockDim.x) {
        if (group_mask[i]) {
            local_sum += ranks_col[i];
        }
    }

    // Warp reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Cross-warp reduction
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < ((blockDim.x + 31) >> 5))
                         ? warp_sums[threadIdx.x]
                         : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            rank_sums[gene] = val;
        }
    }
}

/**
 * Fused z-score + p-value kernel for "with reference" mode.
 *
 * One thread per gene. n_group and n_ref are scalar (single pair).
 */
__global__ void zscore_pvalue_with_ref_kernel(
    const double* __restrict__ rank_sums,  // (n_genes,)
    const double* __restrict__ tie_corr,   // (n_genes,)
    double* __restrict__ z_out,            // (n_genes,)
    double* __restrict__ p_out,            // (n_genes,)
    const int n_combined, const int n_group, const int n_ref, const int n_genes,
    const bool use_continuity) {
    int gene = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene >= n_genes) return;

    double n1 = (double)n_combined + 1.0;
    double expected = (double)n_group * n1 * 0.5;
    double variance =
        tie_corr[gene] * (double)n_group * (double)n_ref * n1 / 12.0;
    double std_dev = sqrt(variance);

    double diff = rank_sums[gene] - expected;
    if (use_continuity) {
        double sign = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
        double abs_diff = fabs(diff) - 0.5;
        if (abs_diff < 0.0) abs_diff = 0.0;
        diff = sign * abs_diff;
    }

    double z = (std_dev > 0.0) ? (diff / std_dev) : 0.0;
    double p = erfc(fabs(z) * M_SQRT1_2);

    z_out[gene] = z;
    p_out[gene] = p;
}
