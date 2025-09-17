#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void gram_csr_upper_kernel(const int* indptr, const int* index, const T* data, int nrows,
                                      int ncols, T* out) {
  int row = blockIdx.x;
  int col_offset = threadIdx.x;
  if (row >= nrows) return;

  int start = indptr[row];
  int end = indptr[row + 1];

  for (int idx1 = start; idx1 < end; ++idx1) {
    int index1 = index[idx1];
    T data1 = data[idx1];
    for (int idx2 = idx1 + col_offset; idx2 < end; idx2 += blockDim.x) {
      int index2 = index[idx2];
      T data2 = data[idx2];
      atomicAdd(&out[(size_t)index1 * ncols + index2], data1 * data2);
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
__global__ void cov_from_gram_kernel(T* cov_values, const T* gram_matrix, const T* mean_x,
                                     const T* mean_y, int ncols) {
  int rid = blockDim.x * blockIdx.x + threadIdx.x;
  int cid = blockDim.y * blockIdx.y + threadIdx.y;
  if (rid >= ncols || cid >= ncols) return;
  cov_values[rid * ncols + cid] = gram_matrix[rid * ncols + cid] - mean_x[rid] * mean_y[cid];
}

__global__ void check_zero_genes_kernel(const int* indices, int* genes, int nnz, int num_genes) {
  int value = blockIdx.x * blockDim.x + threadIdx.x;
  if (value >= nnz) return;
  int gene_index = indices[value];
  if (gene_index >= 0 && gene_index < num_genes) {
    atomicAdd(&genes[gene_index], 1);
  }
}
