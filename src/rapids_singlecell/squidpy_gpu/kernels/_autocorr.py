"""
GPU kernels for spatial autocorrelation (Moran's I and Geary's C).

All kernels are templated on dtype using {0} placeholder for cuda_kernel_factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cuml.common.kernel_utils import cuda_kernel_factory

if TYPE_CHECKING:
    import numpy as np

# =============================================================================
# Moran's I kernels
# =============================================================================

_morans_I_num_dense_kernel = r"""
(const {0}* data_centered, const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind,
const {0}* adj_matrix_data, {0}* num, int n_samples, int n_features) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_samples || f >= n_features) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        {0} edge_weight = adj_matrix_data[k];
        {0} product = data_centered[i * n_features + f] * data_centered[j * n_features + f];
        atomicAdd(&num[f], edge_weight * product);
    }
}
"""

_morans_I_num_sparse_kernel = r"""
(const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind, const {0}* adj_matrix_data,
            const int* data_row_ptr, const int* data_col_ind, const {0}* data_values,
            const int n_samples, const int n_features, const {0}* mean_array,
            {0}* num) {
    int i = blockIdx.x;

    if (i >= n_samples) {
        return;
    }
    int numThreads = blockDim.x;
    int threadid = threadIdx.x;

    // Create cache
    __shared__ {0} cell1[3072];
    __shared__ {0} cell2[3072];

    int numruns = (n_features + 3072 - 1) / 3072;

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        {0} edge_weight = adj_matrix_data[k];

        int cell1_start = data_row_ptr[i];
        int cell1_stop = data_row_ptr[i+1];

        int cell2_start = data_row_ptr[j];
        int cell2_stop = data_row_ptr[j+1];

        for(int batch_runner = 0; batch_runner < numruns; batch_runner++){
            // Set cache to 0
            for (int idx = threadid; idx < 3072; idx += numThreads) {
                cell1[idx] = ({0})(0.0);
                cell2[idx] = ({0})(0.0);
            }
            __syncthreads();
            int batch_start = 3072 * batch_runner;
            int batch_end = 3072 * (batch_runner + 1);

            // Densify sparse into cache
            for (int cell1_idx = cell1_start+ threadid; cell1_idx < cell1_stop;cell1_idx+=numThreads) {
                int gene_id = data_col_ind[cell1_idx];
                if (gene_id >= batch_start && gene_id < batch_end){
                    cell1[gene_id % 3072] = data_values[cell1_idx];
                }
            }
            __syncthreads();
            for (int cell2_idx = cell2_start+threadid; cell2_idx < cell2_stop;cell2_idx+=numThreads) {
                int gene_id = data_col_ind[cell2_idx];
                if (gene_id >= batch_start && gene_id < batch_end){
                    cell2[gene_id % 3072] = data_values[cell2_idx];
                }
            }
            __syncthreads();

            // Calc num
            for(int gene = threadid; gene < 3072; gene+= numThreads){
                int global_gene_index = batch_start + gene;
                if (global_gene_index < n_features) {
                    {0} product = (cell1[gene] - mean_array[global_gene_index]) * (cell2[gene]-mean_array[global_gene_index]);
                    atomicAdd(&num[global_gene_index], edge_weight * product);
                }
            }
        }
    }
}
"""

_pre_den_sparse_kernel = r"""
(const int* data_col_ind, const {0}* data_values, int nnz,
                    const {0}* mean_array,
                    {0}* den, int* counter) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= nnz){
            return;
        }

        int geneidx = data_col_ind[i];
        {0} value = data_values[i] - mean_array[geneidx];
        atomicAdd(&counter[geneidx], 1);
        atomicAdd(&den[geneidx], value*value);
}
"""

# =============================================================================
# Geary's C kernels
# =============================================================================

_gearys_C_num_dense_kernel = r"""
(const {0}* data,
const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind, const {0}* adj_matrix_data,
{0}* num, int n_samples, int n_features) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_samples || f >= n_features) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        {0} edge_weight = adj_matrix_data[k];
        {0} diff_sq = (data[i * n_features + f] - data[j * n_features + f]) * (data[i * n_features + f] - data[j * n_features + f]);
        atomicAdd(&num[f], edge_weight * diff_sq);
    }
}
"""

_gearys_C_num_sparse_kernel = r"""
(const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind, const {0}* adj_matrix_data,
                    const int* data_row_ptr, const int* data_col_ind, const {0}* data_values,
                    const int n_samples, const int n_features,
                    {0}* num) {
    int i = blockIdx.x;
    int numThreads = blockDim.x;
    int threadid = threadIdx.x;

    // Create cache
    __shared__ {0} cell1[3072];
    __shared__ {0} cell2[3072];

    int numruns = (n_features + 3072 - 1) / 3072;

    if (i >= n_samples) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        {0} edge_weight = adj_matrix_data[k];

        int cell1_start = data_row_ptr[i];
        int cell1_stop = data_row_ptr[i+1];

        int cell2_start = data_row_ptr[j];
        int cell2_stop = data_row_ptr[j+1];

        for(int batch_runner = 0; batch_runner < numruns; batch_runner++){
            // Set cache to 0
            for (int idx = threadid; idx < 3072; idx += numThreads) {
                cell1[idx] = ({0})(0.0);
                cell2[idx] = ({0})(0.0);
            }
            __syncthreads();
            int batch_start = 3072 * batch_runner;
            int batch_end = 3072 * (batch_runner + 1);

            // Densify sparse into cache
            for (int cell1_idx = cell1_start+ threadid; cell1_idx < cell1_stop;cell1_idx+=numThreads) {
                int gene_id = data_col_ind[cell1_idx];
                if (gene_id >= batch_start && gene_id < batch_end){
                    cell1[gene_id % 3072] = data_values[cell1_idx];
                }
            }
            __syncthreads();
            for (int cell2_idx = cell2_start+threadid; cell2_idx < cell2_stop;cell2_idx+=numThreads) {
                int gene_id = data_col_ind[cell2_idx];
                if (gene_id >= batch_start && gene_id < batch_end){
                    cell2[gene_id % 3072] = data_values[cell2_idx];
                }
            }
            __syncthreads();

            // Calc num
            for(int gene = threadid; gene < 3072; gene+= numThreads){
                int global_gene_index = batch_start + gene;
                if (global_gene_index < n_features) {
                    {0} diff_sq = (cell1[gene] - cell2[gene]) * (cell1[gene] - cell2[gene]);
                    atomicAdd(&num[global_gene_index], edge_weight * diff_sq);
                }
            }
        }
    }
}
"""


# =============================================================================
# Factory functions
# =============================================================================


def get_morans_I_num_dense_kernel(dtype: np.dtype):
    """Get Moran's I numerator kernel for dense data."""
    return cuda_kernel_factory(
        _morans_I_num_dense_kernel,
        (dtype,),
        "morans_I_num_dense_kernel",
    )


def get_morans_I_num_sparse_kernel(dtype: np.dtype):
    """Get Moran's I numerator kernel for sparse data."""
    return cuda_kernel_factory(
        _morans_I_num_sparse_kernel,
        (dtype,),
        "morans_I_num_sparse_kernel",
    )


def get_pre_den_sparse_kernel(dtype: np.dtype):
    """Get denominator pre-calculation kernel for sparse data."""
    return cuda_kernel_factory(
        _pre_den_sparse_kernel,
        (dtype,),
        "pre_den_sparse_kernel",
    )


def get_gearys_C_num_dense_kernel(dtype: np.dtype):
    """Get Geary's C numerator kernel for dense data."""
    return cuda_kernel_factory(
        _gearys_C_num_dense_kernel,
        (dtype,),
        "gearys_C_num_dense_kernel",
    )


def get_gearys_C_num_sparse_kernel(dtype: np.dtype):
    """Get Geary's C numerator kernel for sparse data."""
    return cuda_kernel_factory(
        _gearys_C_num_sparse_kernel,
        (dtype,),
        "gearys_C_num_sparse_kernel",
    )
