from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

sparse_sparse_aggr_kernel = r"""
    (const int *indptr, const int *index,const {0} *data,
    int* counts,double* sums, double* means, double* vars, int* cats, double* numcells,bool* mask,int n_cells, int n_genes){
    int cell = blockIdx.x;
    if(cell >= n_cells || !mask[cell]){
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

dense_aggr_kernel = r"""
    (const {0} *data, int* counts, double* sums, double* means, double* vars,
    int* cats, double* numcells, bool* mask, int n_cells, int n_genes){
    int cell = blockIdx.x;
    if(cell >= n_cells || !mask[cell]){
        return;
    }

    int group = cats[cell];
    double major = numcells[group];

    for (int gene = threadIdx.x; gene < n_genes; gene += blockDim.x){
        double value = (double)data[cell * n_genes + gene];
        if (value != 0){
            atomicAdd(&sums[group * n_genes + gene], value);
            atomicAdd(&counts[group * n_genes + gene], 1);
            atomicAdd(&means[group * n_genes + gene], value / major);
            atomicAdd(&vars[group * n_genes + gene], value * value / major);;
        }
    }
}
"""


def _get_aggr_sparse_kernel(dtype):
    return cuda_kernel_factory(
        sparse_sparse_aggr_kernel, (dtype,), "sparse_sparse_aggr_kernel"
    )


def _get_aggr_dense_kernel(dtype):
    return cuda_kernel_factory(dense_aggr_kernel, (dtype,), "dense_aggr_kernel")
