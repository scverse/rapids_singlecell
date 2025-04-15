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
