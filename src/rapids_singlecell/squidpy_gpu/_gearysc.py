from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy import sparse

from ._moransi import pre_den_calc_sparse

kernel_gearys_C_num_dense = r"""
extern "C" __global__ void gearys_C_num_dense(const float* data,
const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind, const float* adj_matrix_data,
float* num, int n_samples, int n_features) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_samples || f >= n_features) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        float edge_weight = adj_matrix_data[k];
        float diff_sq = (data[i * n_features + f] - data[j * n_features + f]) * (data[i * n_features + f] - data[j * n_features + f]);
        atomicAdd(&num[f], edge_weight * diff_sq);
    }
}
"""
kernel_gearys_C_num_sparse = r"""
extern "C" __global__
void gearys_C_num_sparse(const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind, const float* adj_matrix_data,
                    const int* data_row_ptr, const int* data_col_ind, const float* data_values,
                    const int n_samples, const int n_features,
                    float* num) {
    int i = blockIdx.x;
    int numThreads = blockDim.x;
    int threadid = threadIdx.x;

    // Create cache
    __shared__ float cell1[3072];
    __shared__ float cell2[3072];

    int numruns = (n_features + 3072 - 1) / 3072;

    if (i >= n_samples) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        float edge_weight = adj_matrix_data[k];

        int cell1_start = data_row_ptr[i];
        int cell1_stop = data_row_ptr[i+1];

        int cell2_start = data_row_ptr[j];
        int cell2_stop = data_row_ptr[j+1];

        for(int batch_runner = 0; batch_runner < numruns; batch_runner++){
            // Set cache to 0
            for (int idx = threadid; idx < 3072; idx += numThreads) {
                cell1[idx] = 0.0f;
                cell2[idx] = 0.0f;
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
                    float diff_sq = (cell1[gene] - cell2[gene]) * (cell1[gene] - cell2[gene]);
                    atomicAdd(&num[global_gene_index], edge_weight * diff_sq);
                }
            }
        }
    }
}
"""


def _gearys_C_cupy_dense(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    # Calculate the numerator for Geary's C
    num = cp.zeros(n_features, dtype=cp.float32)
    num_kernel = cp.RawKernel(kernel_gearys_C_num_dense, "gearys_C_num_dense")

    block_size = 8
    fg = int(math.ceil(n_features / block_size))
    sg = int(math.ceil(n_samples / block_size))
    grid_size = (fg, sg, 1)
    num_kernel(
        grid_size,
        (block_size, block_size, 1),
        (
            data,
            adj_matrix_cupy.indptr,
            adj_matrix_cupy.indices,
            adj_matrix_cupy.data,
            num,
            n_samples,
            n_features,
        ),
    )
    # Calculate the denominator for Geary's C
    gene_mean = data.mean(axis=0).ravel()
    preden = cp.sum((data - gene_mean) ** 2, axis=0)
    den = 2 * adj_matrix_cupy.sum() * preden

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=data.dtype)

        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_kernel(
                grid_size,
                (block_size, block_size, 1),
                (
                    data,
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    num_permuted,
                    n_samples,
                    n_features,
                ),
            )
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy_sparse(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    # Calculate the numerator for Geary's C
    num = cp.zeros(n_features, dtype=cp.float32)
    num_kernel = cp.RawKernel(kernel_gearys_C_num_sparse, "gearys_C_num_sparse")

    n_samples, n_features = data.shape
    sg = n_samples
    # Launch the kernel
    num_kernel(
        (sg,),
        (1024,),
        (
            adj_matrix_cupy.indptr,
            adj_matrix_cupy.indices,
            adj_matrix_cupy.data,
            data.indptr,
            data.indices,
            data.data,
            n_samples,
            n_features,
            num,
        ),
    )
    # Calculate the denominator for Geary's C
    means = data.mean(axis=0).ravel()
    den = cp.zeros(n_features, dtype=cp.float32)
    counter = cp.zeros(n_features, dtype=cp.int32)
    block_den = math.ceil(data.nnz / 32)
    pre_den_kernel = cp.RawKernel(pre_den_calc_sparse, "pre_den_sparse_kernel")

    pre_den_kernel(
        (block_den,), (32,), (data.indices, data.data, data.nnz, means, den, counter)
    )
    counter = n_samples - counter
    den += counter * means**2
    den *= 2 * adj_matrix_cupy.sum()

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=data.dtype)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_kernel(
                (sg,),
                (1024,),
                (
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    data.indptr,
                    data.indices,
                    data.data,
                    n_samples,
                    n_features,
                    num_permuted,
                ),
            )
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy(data, adj_matrix_cupy, n_permutations=100):
    if sparse.isspmatrix_csr(data):
        return _gearys_C_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _gearys_C_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
