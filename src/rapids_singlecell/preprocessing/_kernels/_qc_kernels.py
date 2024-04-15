from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_sparse_qc_kernel_csc = r"""
    (const int *indptr,const int *index,const {0} *data,
        {0}* sums_cells, {0}* sums_genes,
        int* cell_ex, int* gene_ex,
        int n_genes) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene+1];

        {0} sums_genes_i = 0;
        int gene_ex_i = 0;
        for(int cell = start_idx; cell < stop_idx; cell++){
            {0} value = data[cell];
            int cell_number = index[cell];
            sums_genes_i += value;
            atomicAdd(&sums_cells[cell_number], value);
            gene_ex_i += 1;
            atomicAdd(&cell_ex[cell_number], 1);
        }
        sums_genes[gene] = sums_genes_i;
        gene_ex[gene] = gene_ex_i;
    }
"""

_sparse_qc_kernel_csr = r"""
    (const int *indptr,const int *index,const {0} *data,
        {0}* sums_cells, {0}* sums_genes,
        int* cell_ex, int* gene_ex,
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
            atomicAdd(&sums_genes[gene_number], value);
            sums_cells_i += value;
            atomicAdd(&gene_ex[gene_number], 1);
            cell_ex_i += 1;
        }
        sums_cells[cell] = sums_cells_i;
        cell_ex[cell] = cell_ex_i;
    }
"""

_sparse_qc_kernel_dense = r"""
    (const {0} *data,
        {0}* sums_cells, {0}* sums_genes,
        int* cell_ex, int* gene_ex,
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
            atomicAdd(&sums_cells[cell], value);
            atomicAdd(&gene_ex[gene], 1);
            atomicAdd(&cell_ex[cell], 1);
        }
    }
"""

_sparse_qc_kernel_csc_sub = r"""
    (const int *indptr,const int *index,const {0} *data,
        {0}* sums_cells, bool* mask,
        int n_genes) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        if(mask[gene] == false){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene+1];

        for(int cell = start_idx; cell < stop_idx; cell++){
            int cell_number = index[cell];
            atomicAdd(&sums_cells[cell_number], data[cell]);
        }
    }
"""

_sparse_qc_kernel_csr_sub = r"""
    (const int *indptr,const int *index,const {0} *data,
        {0}* sums_cells, bool* mask,
        int n_cells) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        if(cell >= n_cells){
            return;
        }
        int start_idx = indptr[cell];
        int stop_idx = indptr[cell+1];

        {0} sums_cells_i = 0;
        for(int gene = start_idx; gene < stop_idx; gene++){
            int gene_number = index[gene];
            if (mask[gene_number]==true){
                sums_cells_i += data[gene];

            }
        sums_cells[cell] = sums_cells_i;
        }
    }
"""

_sparse_qc_kernel_dense_sub = r"""
    (const {0} *data,
        {0}* sums_cells, bool *mask,
        int n_cells, int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >=n_genes){
            return;
        }
        if(mask[gene] == false){
            return;
        }

        long long int index = static_cast<long long int>(cell) * n_genes + gene;
        atomicAdd(&sums_cells[cell], data[index]);

    }
"""


def _sparse_qc_csc(dtype):
    return cuda_kernel_factory(_sparse_qc_kernel_csc, (dtype,), "_sparse_qc_kernel_csc")


def _sparse_qc_csr(dtype):
    return cuda_kernel_factory(_sparse_qc_kernel_csr, (dtype,), "_sparse_qc_kernel_csr")


def _sparse_qc_dense(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_dense, (dtype,), "_sparse_qc_kernel_dense"
    )


def _sparse_qc_csc_sub(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_csc_sub, (dtype,), "_sparse_qc_kernel_csc_sub"
    )


def _sparse_qc_csr_sub(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_csr_sub, (dtype,), "_sparse_qc_kernel_csr_sub"
    )


def _sparse_qc_dense_sub(dtype):
    return cuda_kernel_factory(
        _sparse_qc_kernel_dense_sub, (dtype,), "_sparse_qc_kernel_dense_sub"
    )
