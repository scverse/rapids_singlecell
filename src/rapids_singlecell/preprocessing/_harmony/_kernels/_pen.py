from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

pen_kernel = r"""
(
    {0}* __restrict__ R,
    const {0}* __restrict__ penalty,
    const int* __restrict__ cats,
    const size_t n_rows,
    const size_t n_cols
)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = n_rows * n_cols;
    if (i >= N) return;

    size_t row = i / n_cols;
    size_t col = i % n_cols;

    int cat = cats[row];
    {0} scale = penalty[(size_t)cat * n_cols + col];
    R[i] *= scale;
}
"""


def _get_pen_kernel(dtype):
    return cuda_kernel_factory(pen_kernel, (dtype,), "pen_kernel")
