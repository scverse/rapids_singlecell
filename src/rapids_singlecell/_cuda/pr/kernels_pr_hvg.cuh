#pragma once

#include <cuda_runtime.h>

// Compute column sums (sums_genes) and row sums (sums_cells) from CSC sparse matrix
// One thread per column (gene), atomicAdd for row sums
template <typename T>
__global__ void sparse_sum_csc_kernel(const int* __restrict__ indptr, const int* __restrict__ index,
                                      const T* __restrict__ data, T* __restrict__ sums_genes,
                                      T* __restrict__ sums_cells, int n_genes) {
  int gene = blockDim.x * blockIdx.x + threadIdx.x;
  if (gene >= n_genes) {
    return;
  }
  int start = indptr[gene];
  int stop = indptr[gene + 1];
  T col_sum = (T)0;
  for (int i = start; i < stop; ++i) {
    T val = data[i];
    col_sum += val;
    atomicAdd(&sums_cells[index[i]], val);
  }
  sums_genes[gene] = col_sum;
}

template <typename T>
__global__ void csc_hvg_res_kernel(const int* __restrict__ indptr, const int* __restrict__ index,
                                   const T* __restrict__ data, const T* __restrict__ sums_genes,
                                   const T* __restrict__ sums_cells, T* __restrict__ residuals,
                                   const T inv_sum_total, const T clip, const T inv_theta,
                                   int n_genes, int n_cells) {
  int gene = blockDim.x * blockIdx.x + threadIdx.x;
  if (gene >= n_genes) {
    return;
  }
  int start = indptr[gene];
  int stop = indptr[gene + 1];

  int sparse_idx = start;
  T var_sum = (T)0;
  T sum_clipped_res = (T)0;
  // first pass to compute mean of clipped residuals per gene
  for (int cell = 0; cell < n_cells; ++cell) {
    T mu = sums_genes[gene] * sums_cells[cell] * inv_sum_total;
    T value = (T)0;
    if (sparse_idx < stop && index[sparse_idx] == cell) {
      value = data[sparse_idx];
      ++sparse_idx;
    }
    T mu_sum = value - mu;
    T clipped_res = mu_sum / sqrt(mu + mu * mu * inv_theta);
    if (clipped_res < -clip) clipped_res = -clip;
    if (clipped_res > clip) clipped_res = clip;
    sum_clipped_res += clipped_res;
  }
  T mean_clipped_res = sum_clipped_res / n_cells;

  // second pass for variance
  sparse_idx = start;
  for (int cell = 0; cell < n_cells; ++cell) {
    T mu = sums_genes[gene] * sums_cells[cell] * inv_sum_total;
    T value = (T)0;
    if (sparse_idx < stop && index[sparse_idx] == cell) {
      value = data[sparse_idx];
      ++sparse_idx;
    }
    T mu_sum = value - mu;
    T clipped_res = mu_sum / sqrt(mu + mu * mu * inv_theta);
    if (clipped_res < -clip) clipped_res = -clip;
    if (clipped_res > clip) clipped_res = clip;
    T diff = clipped_res - mean_clipped_res;
    var_sum += diff * diff;
  }
  residuals[gene] = var_sum / n_cells;
}

template <typename T>
__global__ void dense_hvg_res_kernel(const T* __restrict__ data, const T* __restrict__ sums_genes,
                                     const T* __restrict__ sums_cells, T* __restrict__ residuals,
                                     const T inv_sum_total, const T clip, const T inv_theta,
                                     int n_genes, int n_cells) {
  int gene = blockDim.x * blockIdx.x + threadIdx.x;
  if (gene >= n_genes) {
    return;
  }
  T var_sum = (T)0;
  T sum_clipped_res = (T)0;
  for (int cell = 0; cell < n_cells; ++cell) {
    long long res_index = static_cast<long long>(gene) * n_cells + cell;
    T mu = sums_genes[gene] * sums_cells[cell] * inv_sum_total;
    T value = data[res_index];
    T mu_sum = value - mu;
    T clipped_res = mu_sum / sqrt(mu + mu * mu * inv_theta);
    if (clipped_res < -clip) clipped_res = -clip;
    if (clipped_res > clip) clipped_res = clip;
    sum_clipped_res += clipped_res;
  }
  T mean_clipped_res = sum_clipped_res / n_cells;
  for (int cell = 0; cell < n_cells; ++cell) {
    long long res_index = static_cast<long long>(gene) * n_cells + cell;
    T mu = sums_genes[gene] * sums_cells[cell] * inv_sum_total;
    T value = data[res_index];
    T mu_sum = value - mu;
    T clipped_res = mu_sum / sqrt(mu + mu * mu * inv_theta);
    if (clipped_res < -clip) clipped_res = -clip;
    if (clipped_res > clip) clipped_res = clip;
    T diff = clipped_res - mean_clipped_res;
    var_sum += diff * diff;
  }
  residuals[gene] = var_sum / n_cells;
}
