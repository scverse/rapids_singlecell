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
