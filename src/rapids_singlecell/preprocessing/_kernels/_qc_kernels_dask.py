from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_sparse_qc_kernel_csr_dask_cells = r"""
    (const int *indptr,const int *index,const {0} *data,
            {0}* sums_cells, int* cell_ex,
            int n_cells) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        if(cell >= n_cells){
            return;
        }
        int start_idx = indptr[cell];
        int stop_idx = indptr[cell+1];

        {0} sums_cells_i = 0;
        int cell_ex_i = 0;
        for(int gene = start_idx; gene < stop_idx; gene++){
            {0} value = data[gene];
            int gene_number = index[gene];
            sums_cells_i += value;
            cell_ex_i += 1;
        }
        sums_cells[cell] = sums_cells_i;
        cell_ex[cell] = cell_ex_i;
    }
"""


_sparse_qc_kernel_csr_dask_genes = r"""
        (const int *index,const {0} *data,
            {0}* sums_genes, int* gene_ex,
            int nnz) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx >= nnz){
            return;
        }
        int minor_pos = index[idx];
        atomicAdd(&sums_genes[minor_pos], data[idx]);
        atomicAdd(&gene_ex[minor_pos], 1);
        }
    """

_sparse_qc_kernel_dense_cells = r"""
    (const {0} *data,
        {0}* sums_cells, int* cell_ex,
        int n_cells,int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >=n_genes){
            return;
        }
        long long int index = static_cast<long long int>(cell) * n_genes + gene;
        {0} value = data[index];
        if (value>0.0){
            atomicAdd(&sums_cells[cell], value);
            atomicAdd(&cell_ex[cell], 1);
        }
    }
"""

_sparse_qc_kernel_dense_genes = r"""
    (const {0} *data,
        {0}* sums_genes,int* gene_ex,
        int n_cells,int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >=n_genes){
            return;
        }
        long long int index = static_cast<long long int>(cell) * n_genes + gene;
        {0} value = data[index];
        if (value>0.0){
            atomicAdd(&sums_genes[gene], value);
            atomicAdd(&gene_ex[gene], 1);
        }
    }
"""


def _sparse_qc_csr_dask_cells(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_csr_dask_cells, (dtype,), "_sparse_qc_kernel_csr_dask_cells"
    )


def _sparse_qc_csr_dask_genes(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_csr_dask_genes, (dtype,), "_sparse_qc_kernel_csr_dask_genes"
    )


def _sparse_qc_dense_cells(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_dense_cells, (dtype,), "_sparse_qc_kernel_dense_cells"
    )


def _sparse_qc_dense_genes(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_dense_genes, (dtype,), "_sparse_qc_kernel_dense_genes"
    )
