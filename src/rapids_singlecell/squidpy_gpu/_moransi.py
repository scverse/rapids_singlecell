from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy import sparse

kernel_morans_I_num_dense = r"""
extern "C" __global__
void morans_I_num_dense(const float* data_centered, const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind,
const float* adj_matrix_data, float* num, int n_samples, int n_features) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_samples || f >= n_features) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        float edge_weight = (adj_matrix_data[k]);
        float product = data_centered[i * n_features + f] * data_centered[j * n_features + f];
        atomicAdd(&num[f], edge_weight * product);
    }
}
"""

kernel_morans_I_num_sparse = r"""
extern "C" __global__
void morans_I_num_sparse(const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind, const float* adj_matrix_data,
            const int* data_row_ptr, const int* data_col_ind, const float* data_values,
            const int n_samples, const int n_features, const float* mean_array,
            float* num) {
    int i = blockIdx.x;

    if (i >= n_samples) {
        return;
    }
    int numThreads = blockDim.x;
    int threadid = threadIdx.x;

    // Create cache
    __shared__ float cell1[3072];
    __shared__ float cell2[3072];

    int numruns = (n_features + 3072 - 1) / 3072;

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
                    float product = (cell1[gene] - mean_array[global_gene_index]) * (cell2[gene]-mean_array[global_gene_index]);
                    atomicAdd(&num[global_gene_index], edge_weight * product);
                }
            }
        }
    }
}
"""

pre_den_calc_sparse = r"""
extern "C" __global__
    void pre_den_sparse_kernel(const int* data_col_ind, const float* data_values, int nnz,
                        const float* mean_array,
                        float* den, int* counter) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i >= nnz){
                return;
            }

            int geneidx = data_col_ind[i];
            float value = data_values[i]- mean_array[geneidx];
            atomicAdd(&counter[geneidx], 1);
            atomicAdd(&den[geneidx], value*value);
}
"""


def _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    data_centered_cupy = data - data.mean(axis=0)

    # Calculate the numerator and denominator for Moran's I
    num = cp.zeros(n_features, dtype=cp.float32)
    block_size = 8
    fg = int(math.ceil(n_features / block_size))
    sg = int(math.ceil(n_samples / block_size))
    grid_size = (fg, sg, 1)

    num_kernel = cp.RawKernel(kernel_morans_I_num_dense, "morans_I_num_dense")
    num_kernel(
        grid_size,
        (block_size, block_size, 1),
        (
            data_centered_cupy,
            adj_matrix_cupy.indptr,
            adj_matrix_cupy.indices,
            adj_matrix_cupy.data,
            num,
            n_samples,
            n_features,
        ),
    )

    # Calculate the denominator for Moarn's I
    den = cp.sum(data_centered_cupy**2, axis=0)

    # Calculate Moarn's I
    morans_I = num / den
    # Calculate p-values using permutation tests
    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=cp.float32)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_kernel(
                grid_size,
                (block_size, block_size, 1),
                (
                    data_centered_cupy,
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    num_permuted,
                    n_samples,
                    n_features,
                ),
            )
            morans_I_permutations[p, :] = num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    # Calculate the numerator for Moarn's I
    num = cp.zeros(n_features, dtype=cp.float32)
    num_kernel = cp.RawKernel(kernel_morans_I_num_sparse, "morans_I_num_sparse")
    means = data.mean(axis=0).ravel()

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
            means,
            num,
        ),
    )

    # Calculate the denominator for Moarn's I
    den = cp.zeros(n_features, dtype=cp.float32)
    counter = cp.zeros(n_features, dtype=cp.int32)
    block_den = math.ceil(data.nnz / 32)
    pre_den_kernel = cp.RawKernel(pre_den_calc_sparse, "pre_den_sparse_kernel")

    pre_den_kernel(
        (block_den,), (32,), (data.indices, data.data, data.nnz, means, den, counter)
    )
    counter = n_samples - counter
    den += counter * means**2

    # Calculate Moarn's I
    morans_I = num / den

    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=cp.float32)

        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=data.dtype)
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
                    means,
                    num_permuted,
                ),
            )

            morans_I_permutations[p, :] = num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy(data, adj_matrix_cupy, n_permutations=100):
    if sparse.isspmatrix_csr(data):
        return _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
