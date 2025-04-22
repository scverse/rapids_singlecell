from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

scatter_add_kernel_optimized = r"""(const {0}* __restrict__ v,
                const int* __restrict__ cats,
                long long n_cells,
                long long n_pcs,
                long long switcher,
                {0}* __restrict__ a)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    long long N = n_cells * n_pcs;
    if (i >= N) return;

    long long row = i / n_pcs;  // which cell (row) in R
    long long col = i % n_pcs;  // which column (PC) in R

    long long cat = (long long)cats[row];
    long long out_index  = cat * n_pcs + col;

    // Perform an atomic add on the output array.
    if (switcher==0)atomicAdd(&a[out_index], -v[i]);
    else atomicAdd(&a[out_index], v[i]);
}
"""


def _get_scatter_add_kernel_optimized(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_optimized, (dtype,), "scatter_add_kernel_optimized"
    )


scatter_add_kernel_with_bias = r"""(const {0}* __restrict__ v,
                const int* __restrict__ cats,
                long long n_cells,
                long long n_pcs,
                {0}* __restrict__ a,
                {0}* __restrict__ bias)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    long long N = n_cells * n_pcs;
    if (i >= N) return;

    long long row = i / n_pcs;  // which cell (row) in R
    long long col = i % n_pcs;  // which column (PC) in R

    long long cat = (long long)cats[row]+1;
    long long out_index  = cat * n_pcs + col;

    // Perform an atomic add on the output array.
    atomicAdd(&a[col], v[i]*bias[row]);
    atomicAdd(&a[out_index], v[i]*bias[row]);
}
"""


def _get_scatter_add_kernel_with_bias(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_with_bias, (dtype,), "scatter_add_kernel_with_bias"
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
