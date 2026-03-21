#pragma once

#include <cuda_runtime.h>

template <typename T, typename IdxT>
__global__ void gram_csr_upper_kernel(const IdxT* indptr, const IdxT* index,
                                      const T* data, int nrows, int ncols,
                                      T* out) {
    int row = blockIdx.x;
    int col_offset = threadIdx.x;
    if (row >= nrows) return;

    IdxT start = indptr[row];
    IdxT end = indptr[row + 1];

    for (IdxT idx1 = start; idx1 < end; ++idx1) {
        IdxT index1 = index[idx1];
        T data1 = data[idx1];
        for (IdxT idx2 = idx1 + col_offset; idx2 < end; idx2 += blockDim.x) {
            IdxT index2 = index[idx2];
            T data2 = data[idx2];
            size_t lo = min(index1, index2);
            size_t hi = max(index1, index2);
            atomicAdd(&out[(size_t)lo * ncols + hi], data1 * data2);
        }
    }
}

template <typename T>
__global__ void copy_upper_to_lower_kernel(T* output, int ncols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ncols || col >= ncols) return;
    if (row > col) {
        output[row * ncols + col] = output[col * ncols + row];
    }
}

template <typename T>
__global__ void cov_from_gram_kernel(T* cov_values, const T* gram_matrix,
                                     const T* mean_x, const T* mean_y,
                                     int ncols) {
    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;
    if (rid >= ncols || cid >= ncols) return;
    cov_values[rid * ncols + cid] =
        gram_matrix[rid * ncols + cid] - mean_x[rid] * mean_y[cid];
}

template <typename IdxT>
__global__ void check_zero_genes_kernel(const IdxT* indices, int* genes,
                                        long long nnz, int num_genes) {
    long long value = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (value >= nnz) return;
    int gene_index = static_cast<int>(indices[value]);
    if (gene_index >= 0 && gene_index < num_genes) {
        atomicAdd(&genes[gene_index], 1);
    }
}
