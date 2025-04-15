from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

scatter_add_kernel_optimized = r"""(const {0}* __restrict__ v,
                const int* __restrict__ cats,
                long long n_cells,
                long long n_pcs,
                long long switcher,
                {0}* __restrict__ a,
                {0}* __restrict__ bias)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    long long N = n_cells * n_pcs;
    if (i >= N) return;

    long long row = i / n_pcs;  // which cell (row) in R
    long long col = i % n_pcs;  // which column (PC) in R

    long long cat = (long long)cats[row];
    long long out_index  = cat * n_pcs + col;

    // Perform an atomic add on the output array.
    if (switcher==0)atomicAdd(&a[out_index], -v[i]*bias[row]);
    else atomicAdd(&a[out_index], v[i]*bias[row]);
}
"""


def _get_scatter_add_kernel_optimized(dtype):
    return cuda_kernel_factory(
        scatter_add_kernel_optimized, (dtype,), "scatter_add_kernel_optimized"
    )
