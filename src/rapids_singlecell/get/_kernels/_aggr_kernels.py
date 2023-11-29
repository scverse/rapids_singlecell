from cuml.common.kernel_utils import cuda_kernel_factory

sparse_sparse_aggr_kernel = r"""
    (const int *indptr, const int *index,const {0} *data,
    int* counts,double* sums, double* means, double* vars, int* cats, double* numcells,int n_cells, int n_genes){
    int cell = blockIdx.x;
    if(cell >= n_cells){
        return;
    }
    int cell_start = indptr[cell];
    int cell_end = indptr[cell+1];
    int group = cats[cell];
    double major = numcells[group];
    for (int gene = cell_start+threadIdx.x; gene<cell_end; gene+= blockDim.x){
        int gene_pos = index[gene];
        double value = (double)data[gene];
        atomicAdd(&sums[group*n_genes+gene_pos], value);
        atomicAdd(&counts[group*n_genes+gene_pos], 1);
        atomicAdd(&means[group*n_genes+gene_pos], value/major);
        atomicAdd(&vars[group*n_genes+gene_pos], value*value/major);
    }
}
"""


div_kernel = r"""
(const int *indptr,  {0} *data,
int nrows, {0} *n_cells) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row >= nrows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    {0} div = 1.0/n_cells[row];

    for(int i = start + col; i < end; i += blockDim.x){
        data[i] = data[i] * div;
    }
}
"""

div_kernel2 = r"""
(const int *indptr,  {0} *data,
int nrows, {0} *n_cells) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row >= nrows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    {0} div = n_cells[row]/(n_cells[row]-1);

    for(int i = start + col; i < end; i += blockDim.x){
        data[i] = data[i] * div;
    }
}
"""


def _div_kernel(dtype):
    return cuda_kernel_factory(div_kernel, (dtype,), "div_kernel")


def _div_kernel2(dtype):
    return cuda_kernel_factory(div_kernel2, (dtype,), "div_kernel2")


def _get_aggr_kernel(dtype):
    return cuda_kernel_factory(
        sparse_sparse_aggr_kernel, (dtype,), "sparse_sparse_aggr_kernel"
    )
