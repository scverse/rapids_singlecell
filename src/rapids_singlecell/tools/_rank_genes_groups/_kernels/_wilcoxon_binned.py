"""CUDA histogram kernels for wilcoxon_binned.

All kernels use n_bins+1 bins where bin 0 is reserved for zeros and
bins 1..n_bins form a regular grid over [bin_low, bin_high].

Dense kernel:  Each value is binned inside the kernel. Zero values go
               to bin 0, nonzero values to bins 1..n_bins.

Sparse CSR kernel:
               Only iterates over stored (nonzero) entries and places them
               into bins 1..n_bins. Bin 0 (zero counts) is computed
               outside the kernel as group_size - sum(bins 1..n_bins).

Sparse CSC kernel:
               Same as CSR but one block per gene (column).

Both sparse kernels require non-negative input data.
"""

from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_fused_dense_hist_kernel = r"""
(const {0}* __restrict__ X,
 const int*           __restrict__ gcodes,
 unsigned int*        __restrict__ hist,
 const int n_cells,
 const int n_genes,
 const int n_groups,
 const int n_bins,
 const double bin_low,
 const double inv_bin_width)
{
    const int gene = blockIdx.x;
    const int nbt  = n_bins + 1;
    unsigned int* dst = hist + (long long)gene * n_groups * nbt;

    const {0}* col = X + (long long)gene * n_cells;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
        double val = (double)col[c];
        int grp    = gcodes[c];

        int bin;
        if (val == 0.0) {
            bin = 0;
        } else {
            int raw = (int)((val - bin_low) * inv_bin_width);
            bin = min(max(raw, 0), n_bins - 1) + 1;
        }
        atomicAdd(&dst[grp * nbt + bin], 1u);
    }
}
"""

_fused_csr_hist_kernel = r"""
(const {0}* __restrict__ data,
 const int*           __restrict__ indices,
 const int*           __restrict__ indptr,
 const int*           __restrict__ gcodes,
 unsigned int*        __restrict__ hist,
 const int n_cells,
 const int n_genes,
 const int n_groups,
 const int n_bins,
 const double bin_low,
 const double inv_bin_width,
 const int gene_start)
{
    const int row = blockIdx.x;
    if (row >= n_cells) return;

    const int grp       = gcodes[row];
    const int nbt       = n_bins + 1;
    const int row_start = indptr[row];
    const int row_end   = indptr[row + 1];
    const int gene_stop = gene_start + n_genes;

    // Only bin nonzero entries in [gene_start, gene_stop).
    // Bin 0 (zeros) is filled on the host after the kernel.
    for (int i = row_start + threadIdx.x; i < row_end; i += blockDim.x) {
        const int col = indices[i];
        if (col < gene_start || col >= gene_stop) continue;

        const double val = (double)data[i];
        if (val == 0.0) continue;  // explicit zero skipped

        const int gene = col - gene_start;
        int raw = (int)((val - bin_low) * inv_bin_width);
        int bin = min(max(raw, 0), n_bins - 1) + 1;

        atomicAdd(&hist[(long long)gene * n_groups * nbt + grp * nbt + bin], 1u);
    }
}
"""


_fused_csc_hist_kernel = r"""
(const {0}* __restrict__ data,
 const int*           __restrict__ indices,
 const int*           __restrict__ indptr,
 const int*           __restrict__ gcodes,
 unsigned int*        __restrict__ hist,
 const int n_cells,
 const int n_genes,
 const int n_groups,
 const int n_bins,
 const double bin_low,
 const double inv_bin_width,
 const int gene_start)
{
    const int gene = blockIdx.x;
    if (gene >= n_genes) return;

    const int nbt = n_bins + 1;
    unsigned int* dst = hist + (long long)gene * n_groups * nbt;

    const int col       = gene_start + gene;
    const int col_start = indptr[col];
    const int col_end   = indptr[col + 1];

    // Only bin nonzero entries.
    // Bin 0 (zeros) is filled on the host after the kernel.
    for (int i = col_start + threadIdx.x; i < col_end; i += blockDim.x) {
        const double val = (double)data[i];
        if (val == 0.0) continue;

        const int row = indices[i];
        const int grp = gcodes[row];
        int raw = (int)((val - bin_low) * inv_bin_width);
        int bin = min(max(raw, 0), n_bins - 1) + 1;

        atomicAdd(&dst[grp * nbt + bin], 1u);
    }
}
"""


def _get_dense_hist_kernel(dtype):
    return cuda_kernel_factory(
        _fused_dense_hist_kernel, (dtype,), "_fused_dense_hist_kernel"
    )


def _get_csr_hist_kernel(dtype):
    return cuda_kernel_factory(
        _fused_csr_hist_kernel, (dtype,), "_fused_csr_hist_kernel"
    )


def _get_csc_hist_kernel(dtype):
    return cuda_kernel_factory(
        _fused_csc_hist_kernel, (dtype,), "_fused_csc_hist_kernel"
    )
