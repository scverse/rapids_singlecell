from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

fused_calc_pen_norm_kernel = r"""
(
    const {0}* __restrict__ similarities,  // [n_cells, n_clusters] full matrix
    const {0}* __restrict__ penalty,       // [n_batches, n_clusters]
    const int* __restrict__ cats,          // [block_size] batch categories
    const size_t* __restrict__ idx_in,     // [block_size] indices into similarity matrix
    {0}* __restrict__ R_out,               // [block_size, n_clusters] output
    const {0} term,
    const size_t n_rows,
    const size_t n_cols
)
{
    // One block per row (cell in block), threads handle columns
    size_t row = blockIdx.x;
    if (row >= n_rows) return;

    int cat = cats[row];
    size_t sim_row = idx_in[row];  // Row index in the full similarity matrix

    // Phase 1: Compute exp(term * (1 - sim)) * penalty and accumulate sum
    {0} local_sum = {0}(0);

    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        {0} sim = similarities[sim_row * n_cols + col];
        {0} val = expf(term * ({0}(1) - sim));
        val *= penalty[(size_t)cat * n_cols + col];
        R_out[row * n_cols + col] = val;
        local_sum += val;
    }

    // Phase 2: Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Phase 3: Block-level reduction using shared memory
    __shared__ {0} shared_sum[32];  // One per warp

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    {0} row_sum = {0}(0);
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) >> 5;
        if (threadIdx.x < num_warps) {
            row_sum = shared_sum[threadIdx.x];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }
    }

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum[0];

    // Phase 4: Normalize
    {0} inv_sum = {0}(1) / row_sum;
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        R_out[row * n_cols + col] *= inv_sum;
    }
}
"""


def _get_fused_calc_pen_norm_kernel(dtype):
    return cuda_kernel_factory(
        fused_calc_pen_norm_kernel, (dtype,), "fused_calc_pen_norm"
    )
