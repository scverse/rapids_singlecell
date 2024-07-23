from __future__ import annotations

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
