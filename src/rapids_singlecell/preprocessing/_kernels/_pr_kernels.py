from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_sparse_kernel_sum_csc = r"""
        (const int *indptr,const int *index,const {0} *data,
            {0}* sums_genes, {0}* sums_cells,
            int n_genes) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene+1];

        for(int cell = start_idx; cell < stop_idx; cell++){
            {0} value = data[cell];
            int cell_number = index[cell];
            atomicAdd(&sums_genes[gene], value);
            atomicAdd(&sums_cells[cell_number], value);

        }
    }
    """

_sparse_kernel_norm_res_csc = r"""
        (const int *indptr,const int *index,const {0} *data,
            const {0}* sums_cells,const {0}* sums_genes,
            {0}* residuals ,const {0}* sum_total, const {0}* clip,
            const {0}* theta,const int n_cells,const int n_genes) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene + 1];

        int sparse_idx = start_idx;
        for(int cell = 0; cell < n_cells; cell++){
            {0} mu = sums_genes[gene]*sums_cells[cell]*sum_total[0];
            long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
            if (sparse_idx < stop_idx && index[sparse_idx] == cell){
                residuals[res_index] += data[sparse_idx];
                sparse_idx++;
            }
            residuals[res_index] -= mu;
            residuals[res_index] /= sqrt(mu + mu * mu * theta[0]);
            residuals[res_index]= fminf(fmaxf(residuals[res_index], -clip[0]), clip[0]);
        }
    }
    """

_sparse_kernel_sum_csr = r"""
        (const int *indptr,const int *index,const {0} *data,
            {0}* sums_genes, {0}* sums_cells,
            int n_cells) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        if(cell >= n_cells){
            return;
        }
        int start_idx = indptr[cell];
        int stop_idx = indptr[cell + 1];

        for(int gene = start_idx; gene < stop_idx; gene++){
            {0} value = data[gene];
            int gene_number = index[gene];
            atomicAdd(&sums_genes[gene_number], value);
            atomicAdd(&sums_cells[cell], value);

        }
    }
    """
_sparse_kernel_norm_res_csr = r"""
        (const int * indptr, const int * index, const {0} * data,
            const {0} * sums_cells, const {0} * sums_genes,
            {0} * residuals, const {0} * sum_total, const {0} * clip,
            const {0} * theta, const int n_cells, const int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        if(cell >= n_cells){
            return;
        }
        int start_idx = indptr[cell];
        int stop_idx = indptr[cell + 1];

        int sparse_idx = start_idx;
        for(int gene = 0; gene < n_genes; gene++){
            long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
            {0} mu = sums_genes[gene]*sums_cells[cell]*sum_total[0];
            if (sparse_idx < stop_idx && index[sparse_idx] == gene){
                residuals[res_index] += data[sparse_idx];
                sparse_idx++;
            }
            residuals[res_index] -= mu;
            residuals[res_index] /= sqrt(mu + mu * mu * theta[0]);
            residuals[res_index]= fminf(fmaxf(residuals[res_index], -clip[0]), clip[0]);
        }
    }
    """

_dense_kernel_sum = r"""
        (const {0}* residuals,
                        {0}* sums_cells,{0}* sums_genes,
                        const int n_cells,const int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >= n_genes){
            return;
        }
        long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
        atomicAdd(&sums_genes[gene], residuals[res_index]);
        atomicAdd(&sums_cells[cell], residuals[res_index]);
    }
    """


_kernel_norm_res_dense = r"""
        (const {0}* X,{0}* residuals,
                            const {0}* sums_cells,const {0}* sums_genes,
                            const {0}* sum_total,const {0}* clip,const {0}* theta,
                            const int n_cells, const int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >= n_genes){
            return;
        }

        {0} mu = sums_genes[gene]*sums_cells[cell]*sum_total[0];
        long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
        residuals[res_index] = X[res_index] - mu;
        residuals[res_index] /= sqrt(mu + mu * mu * theta[0]);
        residuals[res_index]= fminf(fmaxf(residuals[res_index], -clip[0]), clip[0]);
    }
    """


def _sparse_sum_csc(dtype):
    return cuda_kernel_factory(
        _sparse_kernel_sum_csc, (dtype,), "_sparse_kernel_sum_csc"
    )


def _sparse_norm_res_csc(dtype):
    return cuda_kernel_factory(
        _sparse_kernel_norm_res_csc, (dtype,), "_sparse_kernel_norm_res_csc"
    )


def _sparse_sum_csr(dtype):
    return cuda_kernel_factory(
        _sparse_kernel_sum_csr, (dtype,), "_sparse_kernel_sum_csr"
    )


def _sparse_norm_res_csr(dtype):
    return cuda_kernel_factory(
        _sparse_kernel_norm_res_csr, (dtype,), "_sparse_kernel_norm_res_csr"
    )


def _sum_dense(dtype):
    return cuda_kernel_factory(_dense_kernel_sum, (dtype,), "_dense_kernel_sum")


def _norm_res_dense(dtype):
    return cuda_kernel_factory(
        _kernel_norm_res_dense, (dtype,), "_kernel_norm_res_dense"
    )


# PR HVG

_csc_hvg_res_kernel = r"""
    (const int *indptr,const int *index,const {0} *data,
        const {0}* sums_genes,const {0}* sums_cells,
        {0}* residuals ,{0}* sum_total,{0}* clip,{0}* theta,int n_genes, int n_cells) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene + 1];

        int sparse_idx = start_idx;
        {0} var_sum = 0.0;
        {0} sum_clipped_res = 0.0;
        for(int cell = 0; cell < n_cells; cell++){
            {0} mu = sums_genes[gene]*sums_cells[cell]/sum_total[0];
            {0} value = 0.0;
            if (sparse_idx < stop_idx && index[sparse_idx] == cell){
                value = data[sparse_idx];
                sparse_idx++;
            }
            {0} mu_sum = value - mu;
            {0} pre_res =  mu_sum / sqrt(mu + mu * mu / theta[0]);
            {0} clipped_res = fminf(fmaxf(pre_res, -clip[0]), clip[0]);
            sum_clipped_res += clipped_res;
        }

        {0} mean_clipped_res = sum_clipped_res / n_cells;
        sparse_idx = start_idx;
        for(int cell = 0; cell < n_cells; cell++){
            {0} mu = sums_genes[gene]*sums_cells[cell]/sum_total[0];
            {0} value = 0.0;
            if (sparse_idx < stop_idx && index[sparse_idx] == cell){
                value = data[sparse_idx];
                sparse_idx++;
            }
            {0} mu_sum = value - mu;
            {0} pre_res =  mu_sum / sqrt(mu + mu * mu / theta[0]);
            {0} clipped_res = fminf(fmaxf(pre_res, -clip[0]), clip[0]);
            {0} diff = clipped_res - mean_clipped_res;
            var_sum += diff * diff;
        }
        residuals[gene] = var_sum / n_cells;
    }

    """


def _csc_hvg_res(dtype):
    return cuda_kernel_factory(_csc_hvg_res_kernel, (dtype,), "_csc_hvg_res_kernel")


_dense_hvg_res_kernel = r"""
    (const {0} *data,
        const {0}* sums_genes,const {0}* sums_cells,
        {0}* residuals ,{0}* sum_total,{0}* clip,{0}* theta,int n_genes, int n_cells) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }

        {0} var_sum = 0.0;
        {0} sum_clipped_res = 0.0;
        for(int cell = 0; cell < n_cells; cell++){
            long long int res_index = static_cast<long long int>(gene) * n_cells + cell;
            {0} mu = sums_genes[gene]*sums_cells[cell]/sum_total[0];
            {0} value = data[res_index];
            {0} mu_sum = value - mu;
            {0} pre_res =  mu_sum / sqrt(mu + mu * mu / theta[0]);
            {0} clipped_res = fminf(fmaxf(pre_res, -clip[0]), clip[0]);
            sum_clipped_res += clipped_res;
        }

        {0} mean_clipped_res = sum_clipped_res / n_cells;
        for(int cell = 0; cell < n_cells; cell++){
            long long int res_index = static_cast<long long int>(gene) * n_cells + cell;
            {0} mu = sums_genes[gene]*sums_cells[cell]/sum_total[0];
            {0} value = data[res_index];
            {0} mu_sum = value - mu;
            {0} pre_res =  mu_sum / sqrt(mu + mu * mu / theta[0]);
            {0} clipped_res = fminf(fmaxf(pre_res, -clip[0]), clip[0]);
            {0} diff = clipped_res - mean_clipped_res;
            var_sum += diff * diff;
        }
        residuals[gene] = var_sum / n_cells;
    }
    """


def _dense_hvg_res(dtype):
    return cuda_kernel_factory(_dense_hvg_res_kernel, (dtype,), "_dense_hvg_res_kernel")
