from __future__ import annotations

import cupy as cp
from cuml.common.kernel_utils import cuda_kernel_factory

cov_kernel_str = r"""
({0} *cov_values, {0} *gram_matrix, {0} *mean_x, {0} *mean_y, int n_cols) {

    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;

    if(rid >= n_cols || cid >= n_cols) return;

    cov_values[rid * n_cols + cid] = \
        gram_matrix[rid * n_cols + cid] - mean_x[rid] * mean_y[cid];
}
"""

gramm_kernel_csr = r"""
(const int *indptr, const int *index, {0} *data, int nrows, int ncols, {0} *out) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row >= nrows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    for (int idx1 = start; idx1 < end; idx1++){
        int index1 = index[idx1];
        {0} data1 = data[idx1];
        for(int idx2 = idx1 + col; idx2 < end; idx2 += blockDim.x){
            int index2 = index[idx2];
            {0} data2 = data[idx2];
            atomicAdd(&out[index1 * ncols + index2], data1 * data2);
        }
    }
}
"""


copy_kernel = r"""
({0} *out, int ncols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ncols || col >= ncols) return;

    if (row > col) {
        out[row * ncols + col] = out[col * ncols + row];
    }
}
"""
check_zero_genes = r"""
extern "C" __global__ void check_zero_genes(const int* indices, int* genes, int nnz) {
    int value = blockIdx.x * blockDim.x + threadIdx.x;
    if(value >= nnz){
        return;
    }
    atomicAdd(&genes[indices[value]], 1);

}
"""
_zero_genes_kernel = cp.RawKernel(check_zero_genes, "check_zero_genes")


def _cov_kernel(dtype):
    return cuda_kernel_factory(cov_kernel_str, (dtype,), "cov_kernel")


def _gramm_kernel_csr(dtype):
    return cuda_kernel_factory(gramm_kernel_csr, (dtype,), "gramm_kernel_csr")


def _copy_kernel(dtype):
    return cuda_kernel_factory(copy_kernel, (dtype,), "copy_kernel")
