from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

out_kernel_code = r"""
({0}* __restrict__ E,
    const {0}* __restrict__ Pr_b,
    const {0}* __restrict__ R_sum,
    long long n_cats,
    long long n_pcs,
    long long switcher)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    long long N = n_cats * n_pcs;
    if (i >= N) return;

    // Determine row and column from the flattened index.
    long long row = i / n_pcs;  // which cell (row) in R
    long long col = i % n_pcs;  // which column (PC) in R

    if (switcher==0) E[i] -= (Pr_b[row] * R_sum[col]);
    else E[i] += (Pr_b[row] * R_sum[col]);
}
"""


def _get_outer_kernel(dtype):
    return cuda_kernel_factory(out_kernel_code, (dtype,), "outer_kernel")


harmony_correction_kernel_code = r"""
({0}* __restrict__ Z,
    const {0}* __restrict__ W,
    const int* __restrict__ cats,
    const {0}* __restrict__ R,
    long long n_cells,
    long long n_pcs)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_cells * n_pcs) return;

    // Determine row and column from the flattened index
    long long cell_idx = i / n_pcs;  // which cell (row)
    long long pc_idx = i % n_pcs;    // which PC (column)

    // Get the category/batch for this cell
    int cat = cats[cell_idx];

    // Calculate correction term: (W[1:][cats] + W[0]) * R[:, k]
    {0} correction = W[(cat + 1)*n_pcs + pc_idx] * R[cell_idx];

    // Apply correction: Z -= correction
    Z[i] -= correction;
}
"""


def _get_harmony_correction_kernel(dtype):
    return cuda_kernel_factory(
        harmony_correction_kernel_code, (dtype,), "harmony_correction_kernel"
    )


_colsum_kernel = r"""
(const {0}* __restrict__ A,
            {0}* __restrict__ out,
            size_t rows,
            size_t cols) {
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < cols; col += gridDim.x) {
        {0} acc = {0}(0);
        for (size_t i = tid; i < rows; i += blockDim.x) {
            acc += A[i * cols + col];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1){
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        static __shared__ {0} s[32];
        if ((threadIdx.x & 31) == 0){
            s[threadIdx.x>>5] = acc;
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            {0} val = (threadIdx.x < (blockDim.x>>5))
                            ? s[threadIdx.x]
                            : {0}(0);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, off);
            }
            if (threadIdx.x == 0) {
                out[col] =val;
            }
        }
    }
}
"""


def _get_colsum_kernel(dtype):
    return cuda_kernel_factory(
        _colsum_kernel,
        (dtype,),
        "_colsum_kernel",
    )


_colsum_atomic_code = r"""
(const {0}* __restrict__ A,
        {0}* __restrict__ out,
        size_t rows,
        size_t cols,
        size_t rows_per_tile) {
    // Block handles one column-tile, processes rows_per_tile rows
    // blockIdx.x = column tile index
    // blockIdx.y = row tile index
    // Uses shared memory to reduce atomics: one per column per block

    // Shared memory for column sums (one per column in tile)
    __shared__ {0} col_sums[32];

    size_t col = blockIdx.x * 32 + threadIdx.y;
    size_t start_row = blockIdx.y * rows_per_tile;
    size_t end_row = min(start_row + rows_per_tile, rows);

    // Initialize shared memory (first warp)
    if (threadIdx.x < 32) {
        col_sums[threadIdx.x] = {0}(0);
    }
    __syncthreads();

    // Each thread accumulates multiple rows
    {0} acc = {0}(0);
    if (col < cols) {
        #pragma unroll 4
        for (size_t row = start_row + threadIdx.x; row < end_row; row += 32) {
            acc += A[row * cols + col];
        }
    }

    // Warp-level reduction across 32 threads (different rows)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }

    // Lane 0 of each warp accumulates into shared memory
    if (threadIdx.x == 0 && threadIdx.y < 32) {
        atomicAdd(&col_sums[threadIdx.y], acc);
    }
    __syncthreads();

    // First warp writes to global memory (one atomic per column)
    if (threadIdx.x == 0 && threadIdx.y < 32) {
        size_t out_col = blockIdx.x * 32 + threadIdx.y;
        if (out_col < cols && col_sums[threadIdx.y] != {0}(0)) {
            atomicAdd(&out[out_col], col_sums[threadIdx.y]);
        }
    }
}
"""


def _get_colsum_atomic_kernel(dtype):
    return cuda_kernel_factory(
        _colsum_atomic_code,
        (dtype,),
        "colsum_atomic",
    )
