from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_expected_zeros_kernel_str = r"""
    (const {0}* scaled_means,
     const {0}* total_counts,
     {0}* expected,
     const int n_genes,
     const int n_cells) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if (gene >= n_genes) return;

        {0} sm = scaled_means[gene];
        {0} sum = ({0})0.0;

        for (int c = 0; c < n_cells; c++) {
            sum += exp(-sm * total_counts[c]);
        }

        expected[gene] = sum / ({0})n_cells;
    }
"""


def _expected_zeros_kernel(dtype):
    return cuda_kernel_factory(
        _expected_zeros_kernel_str, (dtype,), "_expected_zeros_kernel"
    )
