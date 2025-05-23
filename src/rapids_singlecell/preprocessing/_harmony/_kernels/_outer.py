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
        size_t cols) {
    // how many 32-wide column tiles
    size_t tile_cols = (cols + 31) / 32;
    size_t tid      = blockIdx.x;
    size_t tile_r   = tid / tile_cols;
    size_t tile_c   = tid % tile_cols;

    // compute our element coords
    size_t row = tile_r * 32 + threadIdx.x;
    size_t col = tile_c * 32 + threadIdx.y;

    {0} v = {0}(0);
    if (row < rows && col < cols) {
        // coalesced load: all threads in this warp touch
        // col = tile_c*32 + warp_lane in [0..31]
        v = A[row * cols + col];
    }

    // warp‐level sum over the 32 rows in this tile‐column
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }

    // lane 0 of each warp writes one atomicAdd for this column
    if (threadIdx.x == 0 && col < cols) {
        atomicAdd(&out[col], v);
    }
}
"""


def _get_colsum_atomic_kernel(dtype):
    return cuda_kernel_factory(
        _colsum_atomic_code,
        (dtype,),
        "colsum_atomic",
    )
