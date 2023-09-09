import math

import cupy as cp
from cupyx.scipy import sparse

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
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=data.dtype)
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
            # den_permuted = 2 * adj_matrix_permuted.sum() * preden
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy_sparse(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape

    # Calculate the denominator for Geary's C
    gene_mean = data.mean(axis=0).ravel()
    preden = cp.sum((data - gene_mean) ** 2, axis=0)
    den = 2 * adj_matrix_cupy.sum() * preden

    # Calculate the numerator for Geary's C
    data = data.tocsc()
    num = cp.zeros(n_features, dtype=data.dtype)
    block_size = 8
    sg = int(math.ceil(n_samples / block_size))
    num_kernel = cp.RawKernel(kernel_gearys_C_num_dense, "gearys_C_num_dense")
    batchsize = 1000
    n_batches = math.ceil(n_features / batchsize)
    for batch in range(n_batches):
        start_idx = batch * batchsize
        stop_idx = min(batch * batchsize + batchsize, n_features)
        data_block = data[:, start_idx:stop_idx].toarray()
        num_block = cp.zeros(data_block.shape[1], dtype=data.dtype)
        fg = int(math.ceil(data_block.shape[1] / block_size))
        grid_size = (fg, sg, 1)

        num_kernel(
            grid_size,
            (block_size, block_size, 1),
            (
                data_block,
                adj_matrix_cupy.indptr,
                adj_matrix_cupy.indices,
                adj_matrix_cupy.data,
                num_block,
                n_samples,
                data_block.shape[1],
            ),
        )
        num[start_idx:stop_idx] = num_block
        cp.cuda.Stream.null.synchronize()

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=data.dtype)
            for batch in range(n_batches):
                start_idx = batch * batchsize
                stop_idx = min(batch * batchsize + batchsize, n_features)
                data_block = data[:, start_idx:stop_idx].toarray()
                num_block = cp.zeros(data_block.shape[1], dtype=data.dtype)
                fg = int(math.ceil(data_block.shape[1] / block_size))
                grid_size = (fg, sg, 1)

                num_kernel(
                    grid_size,
                    (block_size, block_size, 1),
                    (
                        data_block,
                        adj_matrix_permuted.indptr,
                        adj_matrix_permuted.indices,
                        adj_matrix_permuted.data,
                        num_block,
                        n_samples,
                        data_block.shape[1],
                    ),
                )
                num_permuted[start_idx:stop_idx] = num_block

            # den_permuted = 2 * adj_matrix_permuted.sum() * preden
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
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
