from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

sparse_dense_aggr_kernel = r"""
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

sparse_sparse_aggr_kernel = r"""
    (const int *indptr, const int *index, const {0}* data,
    int* row, int* col, double* ndata,
    int* cats,bool* mask,int n_cells){
    int cell = blockIdx.x;
    if(cell >= n_cells || !mask[cell]){
        return;
    }
    int cell_start = indptr[cell];
    int cell_end = indptr[cell+1];
    int group = cats[cell];
    for (int gene = cell_start+threadIdx.x; gene<cell_end; gene+= blockDim.x){
        int gene_pos = index[gene];
        ndata[gene] = (double)data[gene];
        row[gene] = group;
        col[gene] = gene_pos;
    }
}
"""

sparse_var_kernel = r"""
    (const int *indptr, const int *index,double *data,
    const double *mean_data, double *n_cells, int dof, int n_groups){
    int group = blockIdx.x;
    if(group >= n_groups){
        return;
    }
    int group_start = indptr[group];
    int group_end = indptr[group+1];
    double doffer = n_cells[group]/(n_cells[group]-dof);
    for (int gene = group_start+threadIdx.x; gene<group_end; gene+= blockDim.x){
        double var = data[gene];
        double mean = mean_data[gene]*mean_data[gene];
        var = var- mean;
        data[gene] = var* doffer;
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
        sparse_dense_aggr_kernel, (dtype,), "sparse_dense_aggr_kernel"
    )


def _get_aggr_sparse_sparse_kernel(dtype):
    return cuda_kernel_factory(
        sparse_sparse_aggr_kernel, (dtype,), "sparse_sparse_aggr_kernel"
    )


def _get_sparse_var_kernel(dtype):
    return cuda_kernel_factory(sparse_var_kernel, (dtype,), "sparse_var_kernel")


def _get_aggr_dense_kernel(dtype):
    return cuda_kernel_factory(dense_aggr_kernel, (dtype,), "dense_aggr_kernel")
