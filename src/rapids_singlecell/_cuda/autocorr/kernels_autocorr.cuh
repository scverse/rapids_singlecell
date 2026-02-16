#pragma once

#include <cuda_runtime.h>

// Moran's I - dense numerator
template <typename T>
__global__ void morans_I_num_dense_kernel(const T* __restrict__ data_centered,
                                          const int* __restrict__ adj_row_ptr,
                                          const int* __restrict__ adj_col_ind,
                                          const T* __restrict__ adj_data, T* __restrict__ num,
                                          int n_samples, int n_features) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_samples || f >= n_features) {
    return;
  }
  int k_start = adj_row_ptr[i];
  int k_end = adj_row_ptr[i + 1];
  for (int k = k_start; k < k_end; ++k) {
    int j = adj_col_ind[k];
    T w = adj_data[k];
    T prod = data_centered[i * n_features + f] * data_centered[j * n_features + f];
    atomicAdd(&num[f], w * prod);
  }
}

// Moran's I - sparse numerator
template <typename T>
__global__ void morans_I_num_sparse_kernel(
    const int* __restrict__ adj_row_ptr, const int* __restrict__ adj_col_ind,
    const T* __restrict__ adj_data, const int* __restrict__ data_row_ptr,
    const int* __restrict__ data_col_ind, const T* __restrict__ data_values, int n_samples,
    int n_features, const T* __restrict__ mean_array, T* __restrict__ num) {
  int i = blockIdx.x;
  if (i >= n_samples) {
    return;
  }
  int numThreads = blockDim.x;
  int threadid = threadIdx.x;

  __shared__ T cell1[3072];
  __shared__ T cell2[3072];
  int numruns = (n_features + 3072 - 1) / 3072;
  int k_start = adj_row_ptr[i];
  int k_end = adj_row_ptr[i + 1];
  for (int k = k_start; k < k_end; ++k) {
    int j = adj_col_ind[k];
    T w = adj_data[k];
    int cell1_start = data_row_ptr[i];
    int cell1_stop = data_row_ptr[i + 1];
    int cell2_start = data_row_ptr[j];
    int cell2_stop = data_row_ptr[j + 1];
    for (int run = 0; run < numruns; ++run) {
      for (int idx = threadid; idx < 3072; idx += numThreads) {
        cell1[idx] = T(0);
        cell2[idx] = T(0);
      }
      __syncthreads();
      int batch_start = 3072 * run;
      int batch_end = 3072 * (run + 1);
      for (int a = cell1_start + threadid; a < cell1_stop; a += numThreads) {
        int g = data_col_ind[a];
        if (g >= batch_start && g < batch_end) {
          cell1[g % 3072] = data_values[a];
        }
      }
      __syncthreads();
      for (int b = cell2_start + threadid; b < cell2_stop; b += numThreads) {
        int g = data_col_ind[b];
        if (g >= batch_start && g < batch_end) {
          cell2[g % 3072] = data_values[b];
        }
      }
      __syncthreads();
      for (int gene = threadid; gene < 3072; gene += numThreads) {
        int global_gene = batch_start + gene;
        if (global_gene < n_features) {
          T prod =
              (cell1[gene] - mean_array[global_gene]) * (cell2[gene] - mean_array[global_gene]);
          atomicAdd(&num[global_gene], w * prod);
        }
      }
    }
  }
}

// Geary's C - dense numerator
template <typename T>
__global__ void gearys_C_num_dense_kernel(const T* __restrict__ data,
                                          const int* __restrict__ adj_row_ptr,
                                          const int* __restrict__ adj_col_ind,
                                          const T* __restrict__ adj_data, T* __restrict__ num,
                                          int n_samples, int n_features) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_samples || f >= n_features) {
    return;
  }
  int k_start = adj_row_ptr[i];
  int k_end = adj_row_ptr[i + 1];
  for (int k = k_start; k < k_end; ++k) {
    int j = adj_col_ind[k];
    T w = adj_data[k];
    T diff = data[i * n_features + f] - data[j * n_features + f];
    atomicAdd(&num[f], w * diff * diff);
  }
}

// Geary's C - sparse numerator
template <typename T>
__global__ void gearys_C_num_sparse_kernel(const int* __restrict__ adj_row_ptr,
                                           const int* __restrict__ adj_col_ind,
                                           const T* __restrict__ adj_data,
                                           const int* __restrict__ data_row_ptr,
                                           const int* __restrict__ data_col_ind,
                                           const T* __restrict__ data_values, int n_samples,
                                           int n_features, T* __restrict__ num) {
  int i = blockIdx.x;
  int numThreads = blockDim.x;
  int threadid = threadIdx.x;
  __shared__ T cell1[3072];
  __shared__ T cell2[3072];
  int numruns = (n_features + 3072 - 1) / 3072;
  if (i >= n_samples) {
    return;
  }
  int k_start = adj_row_ptr[i];
  int k_end = adj_row_ptr[i + 1];
  for (int k = k_start; k < k_end; ++k) {
    int j = adj_col_ind[k];
    T w = adj_data[k];
    int cell1_start = data_row_ptr[i];
    int cell1_stop = data_row_ptr[i + 1];
    int cell2_start = data_row_ptr[j];
    int cell2_stop = data_row_ptr[j + 1];
    for (int run = 0; run < numruns; ++run) {
      for (int idx = threadid; idx < 3072; idx += numThreads) {
        cell1[idx] = T(0);
        cell2[idx] = T(0);
      }
      __syncthreads();
      int batch_start = 3072 * run;
      int batch_end = 3072 * (run + 1);
      for (int a = cell1_start + threadid; a < cell1_stop; a += numThreads) {
        int g = data_col_ind[a];
        if (g >= batch_start && g < batch_end) {
          cell1[g % 3072] = data_values[a];
        }
      }
      __syncthreads();
      for (int b = cell2_start + threadid; b < cell2_stop; b += numThreads) {
        int g = data_col_ind[b];
        if (g >= batch_start && g < batch_end) {
          cell2[g % 3072] = data_values[b];
        }
      }
      __syncthreads();
      for (int gene = threadid; gene < 3072; gene += numThreads) {
        int global_gene = batch_start + gene;
        if (global_gene < n_features) {
          T diff = cell1[gene] - cell2[gene];
          atomicAdd(&num[global_gene], w * diff * diff);
        }
      }
    }
  }
}

// Pre-denominator for sparse paths
template <typename T>
__global__ void pre_den_sparse_kernel(const int* __restrict__ data_col_ind,
                                      const T* __restrict__ data_values, int nnz,
                                      const T* __restrict__ mean_array, T* __restrict__ den,
                                      int* __restrict__ counter) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz) {
    return;
  }
  int geneidx = data_col_ind[i];
  T value = data_values[i] - mean_array[geneidx];
  atomicAdd(&counter[geneidx], 1);
  atomicAdd(&den[geneidx], value * value);
}
