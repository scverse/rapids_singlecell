from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

scatter_add_kernel_optimized = r"""(const {0}* __restrict__ v,
                const int* __restrict__ cats,
                size_t n_cells,
                size_t n_pcs,
                size_t switcher,
                {0}* __restrict__ a)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = n_cells * n_pcs;
    if (i >= N) return;

    size_t row = i / n_pcs;  // which cell (row) in R
    size_t col = i % n_pcs;  // which column (PC) in R

    size_t cat = (size_t)cats[row];
    size_t out_index  = cat * n_pcs + col;

    // Perform an atomic add on the output array.
    if (switcher==0)atomicAdd(&a[out_index], -v[i]);
    else atomicAdd(&a[out_index], v[i]);
}
"""


def _get_scatter_add_kernel_optimized(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_optimized, (dtype,), "scatter_add_kernel_optimized"
    )


aggregated_matrix_kernel = r"""({0}* __restrict__ aggregated_matrix,
                const {0}* __restrict__ sum,
                {0}* __restrict__ top_corner,
                int n_batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_batches+1) return;

    if (i == 0) {
        aggregated_matrix[0] = top_corner[0];
    } else {
        aggregated_matrix[i] = sum[i-1];
        aggregated_matrix[(n_batches+1)*i] = sum[i-1];
        aggregated_matrix[(n_batches+1)*i+i] = sum[i-1];
    }
}
"""


def _get_aggregated_matrix_kernel(dtype):
    return cuda_kernel_factory(
        aggregated_matrix_kernel, (dtype,), "aggregated_matrix_kernel"
    )


scatter_add_kernel_with_bias_cat0 = r"""(const {0}* __restrict__ v,
                int n_cells,
                int n_pcs,
                {0}* __restrict__ a,
                const {0}* __restrict__ bias)
{
    using VecPC = {0}2;
    // Each block handles one PC pair and 1/4 of the cells
    int pairs = (n_pcs + 1) / 2;
    int pc_pair = blockIdx.x;
    int eighth = blockIdx.y;

    if (pc_pair >= pairs) return;

    int pc0 = pc_pair * 2;
    int pc1 = pc0 + 1;
    bool has_pc1 = (pc1 < n_pcs);

    {0} acc0 = {0}(0);
    {0} acc1 = {0}(0);

    // Calculate cell range for this block
    int cells_per_eighth = (n_cells + 7) / 8;
    int start_cell = eighth * cells_per_eighth;
    int end_cell = min(start_cell + cells_per_eighth, n_cells);

    // Unroll the main processing loop
    #pragma unroll 4
    for (int i = start_cell + threadIdx.x; i < end_cell; i += blockDim.x) {
        size_t base = size_t(i) * n_pcs + pc0;
        VecPC vv = *(const VecPC*)(v + base);
        {0} bb = __ldg(bias + i);
        acc0 += vv.x * bb;
        if (has_pc1) acc1 += vv.y * bb;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1){
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        if (has_pc1) {
            acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        }
    }

    static __shared__ VecPC s[32];
    if ((threadIdx.x & 31) == 0)
        s[threadIdx.x>>5] = VecPC{acc0, acc1};
    __syncthreads();

    if (threadIdx.x < 32) {
        VecPC val = (threadIdx.x < (blockDim.x>>5))
                        ? s[threadIdx.x]
                        : VecPC{0,0};
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val.x += __shfl_down_sync(0xffffffff, val.x, off);
            val.y += __shfl_down_sync(0xffffffff, val.y, off);
        }
        if (threadIdx.x == 0) {
            // Use atomic to combine results from all quarters
            int out_base = 0 * n_pcs + pc0;  // cat is 0
            atomicAdd(&a[out_base], val.x);
            if (has_pc1) atomicAdd(&a[out_base+1], val.y);
        }
    }
}
"""


def _get_scatter_add_kernel_with_bias_cat0(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_with_bias_cat0,
        (dtype,),
        "scatter_add_kernel_with_bias_cat0",
    )


scatter_add_kernel_with_bias_block = r"""(const {0}* __restrict__ v,
                const int* __restrict__ cat_offsets,
                const int* __restrict__ cell_indices,
                int n_cells,
                int n_pcs,
                int n_batches,
                {0}* __restrict__ a,
                const {0}* __restrict__ bias)
{
    using VecPC = {0}2;
    // Each block handles one (category, PC) combination
    int pairs     = (n_pcs + 1) / 2;
    int block_idx = blockIdx.x;
    if (block_idx >= n_batches*pairs) return;

    int cat     = block_idx / pairs + 1;  // Start from cat=1
    int pc_pair = block_idx % pairs;

    int pc0       = pc_pair*2;
    int pc1       = pc0 + 1;
    bool has_pc1  = (pc1 < n_pcs);

    {0} acc0 = {0}(0);
    {0} acc1 = {0}(0);

    // Get range of cell indices for this category
    int start_idx = cat_offsets[cat-1];
    int end_idx = cat_offsets[cat];

    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        int cell_idx = cell_indices[i];
        size_t in_index = static_cast<size_t>(cell_idx)* n_pcs + pc0;
        VecPC vv = *(const VecPC*)(v + in_index);
        {0} bb = __ldg(bias + cell_idx);
        acc0 += vv.x * bb;
        if (has_pc1) acc1 += vv.y * bb;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1){
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        if (has_pc1) {
            acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        }
    }

    static __shared__ VecPC s[32];
    if ((threadIdx.x & 31) == 0)
        s[threadIdx.x>>5] = VecPC{acc0, acc1};
    __syncthreads();

    if (threadIdx.x < 32) {
        VecPC val = (threadIdx.x < (blockDim.x>>5))
                        ? s[threadIdx.x]
                        : VecPC{0,0};
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val.x += __shfl_down_sync(0xffffffff, val.x, off);
            val.y += __shfl_down_sync(0xffffffff, val.y, off);
        }
        if (threadIdx.x == 0) {
            // write two outputs for this block:
            int out_base = cat*n_pcs + pc0;
            a[out_base]   = val.x;
            if (has_pc1)  a[out_base+1] = val.y;
        }
    }
}
"""


def _get_scatter_add_kernel_with_bias_block(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_with_bias_block,
        (dtype,),
        "scatter_add_kernel_with_bias_block",
    )
