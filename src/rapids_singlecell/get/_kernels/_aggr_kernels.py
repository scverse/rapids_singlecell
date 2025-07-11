from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

sparse_dense_aggr_kernel = r"""
    (const int *indptr, const int *index,const {0} *data,
    double* out, int* cats,bool* mask,
    size_t n_cells, size_t n_genes, size_t n_groups){
    size_t cell = blockIdx.x;
    if(cell >= n_cells || !mask[cell]){
        return;
    }
    int cell_start = indptr[cell];
    int cell_end = indptr[cell+1];
    size_t group = (size_t)cats[cell];
    for (int gene = cell_start+threadIdx.x; gene<cell_end; gene+= blockDim.x){
        size_t gene_pos = (size_t)index[gene];
        double value = (double)data[gene];
        atomicAdd(&out[group*n_genes+gene_pos], value);
        atomicAdd(&out[group*n_genes+gene_pos+n_genes*n_groups], 1);
        atomicAdd(&out[group*n_genes+gene_pos+2*n_genes*n_groups], value*value);
    }
}
"""


sparse_dense_aggr_kernel_csc = r"""
    (const int *indptr, const int *index,const {0} *data,
    double* out, int* cats,bool* mask,
    size_t n_cells, size_t n_genes, size_t n_groups){
    size_t gene = blockIdx.x;
    if(gene >= n_genes){
        return;
    }
    int gene_start = indptr[gene];
    int gene_end = indptr[gene+1];

    for (int cell_idx = gene_start+threadIdx.x; cell_idx<gene_end; cell_idx+= blockDim.x){
        size_t cell = (size_t)index[cell_idx];
        if(!mask[cell]){
            continue;
        }
        size_t group = (size_t)cats[cell];
        double value = (double)data[cell_idx];
        atomicAdd(&out[group*n_genes+gene], value);
        atomicAdd(&out[group*n_genes+gene+n_genes*n_groups], 1);
        atomicAdd(&out[group*n_genes+gene+2*n_genes*n_groups], value*value);
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


dense_aggr_kernel_C = r"""
    (const {0} *data, double* out,
    int* cats, bool* mask, size_t n_cells, size_t n_genes, size_t n_groups){

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t N = n_cells * n_genes;
    while (i < N){
        if (i >= N) return;
        size_t cell = i / n_genes;
        size_t gene = i % n_genes;
        if(mask[cell]){
            size_t group = (size_t) cats[cell];

            double value = (double)data[cell * n_genes + gene];
            if (value != 0){
                atomicAdd(&out[group*n_genes+gene], value);
                atomicAdd(&out[group*n_genes+gene+n_genes*n_groups], 1);
                atomicAdd(&out[group*n_genes+gene+2*n_genes*n_groups], value*value);
            }
        }
        i += stride;
    }
}
"""

dense_aggr_kernel_F = r"""
    (const {0} *data, double* out,
    int* cats, bool* mask, size_t n_cells, size_t n_genes, size_t n_groups){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t N = n_cells * n_genes;
    while (i < N){
        if (i >= N) return;
        size_t cell = i % n_cells;
        size_t gene = i / n_cells;
        if(mask[cell]){
            size_t group = (size_t) cats[cell];

            double value = (double)data[gene * n_cells + cell];
            if (value != 0){
                atomicAdd(&out[group*n_genes+gene], value);
                atomicAdd(&out[group*n_genes+gene+n_genes*n_groups], 1);
                atomicAdd(&out[group*n_genes+gene+2*n_genes*n_groups], value*value);
            }
        }
        i += stride;
    }
}
"""


def _get_aggr_sparse_kernel(dtype):
    return cuda_kernel_factory(
        sparse_dense_aggr_kernel, (dtype,), "sparse_dense_aggr_kernel"
    )


def _get_aggr_sparse_kernel_csc(dtype):
    return cuda_kernel_factory(
        sparse_dense_aggr_kernel_csc, (dtype,), "sparse_dense_aggr_kernel_csc"
    )


def _get_aggr_sparse_sparse_kernel(dtype):
    return cuda_kernel_factory(
        sparse_sparse_aggr_kernel, (dtype,), "sparse_sparse_aggr_kernel"
    )


def _get_sparse_var_kernel(dtype):
    return cuda_kernel_factory(sparse_var_kernel, (dtype,), "sparse_var_kernel")


def _get_aggr_dense_kernel_C(dtype):
    return cuda_kernel_factory(dense_aggr_kernel_C, (dtype,), "dense_aggr_kernel_C")


def _get_aggr_dense_kernel_F(dtype):
    return cuda_kernel_factory(dense_aggr_kernel_F, (dtype,), "dense_aggr_kernel_F")
