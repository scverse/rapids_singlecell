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


scatter_add_kernel_with_bias_block = r"""(const {0}* __restrict__ v,
                const int* __restrict__ cat_offsets,
                const int* __restrict__ cell_indices,
                int n_cells,
                int n_pcs,
                int n_batches,
                {0}* __restrict__ a,
                const {0}* __restrict__ bias)
{
    // Each block handles one (category, PC) combination
    int block_idx = blockIdx.x;
    int total_blocks = (n_batches+1) * n_pcs;

    if (block_idx >= total_blocks) return;

    // Determine category and PC from block index
    int cat = block_idx / n_pcs;
    int pc = block_idx % n_pcs;
    __shared__ {0} sdata[256];

    // Initialize shared memory
    sdata[threadIdx.x] = {0}(0);
    // Run the first row of a
    if(cat == 0){
        for(int i = threadIdx.x; i < n_cells; i += blockDim.x){
        size_t in_index = static_cast<size_t>(i)* n_pcs + pc;
        sdata[threadIdx.x] += v[in_index] * bias[i];
        }
    }
    // Run the rest of the rows of a
    else{
        // Get range of cell indices for this category
        int start_idx = cat_offsets[cat-1];
        int end_idx = cat_offsets[cat];

        for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
            int cell_idx = cell_indices[i];
            size_t in_index = static_cast<size_t>(cell_idx)* n_pcs + pc;
            sdata[threadIdx.x] += v[in_index] * bias[cell_idx];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 256 / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    a[block_idx] =sdata[0];
}
"""


def _get_scatter_add_kernel_with_bias_block(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_with_bias_block,
        (dtype,),
        "scatter_add_kernel_with_bias_block",
    )
