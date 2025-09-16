#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void sum_and_count_dense_kernel(const T* __restrict__ data,
                                           const int* __restrict__ clusters,
                                           T* __restrict__ sum_gt0, int* __restrict__ count_gt0,
                                           int num_rows, int num_cols, int n_cls) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= num_rows || j >= num_cols) return;
  int cluster = clusters[i];
  T value = data[i * num_cols + j];
  if (value > (T)0) {
    atomicAdd(&sum_gt0[j * n_cls + cluster], value);
    atomicAdd(&count_gt0[j * n_cls + cluster], 1);
  }
}

template <typename T>
__global__ void sum_and_count_sparse_kernel(const int* __restrict__ indptr,
                                            const int* __restrict__ index,
                                            const T* __restrict__ data,
                                            const int* __restrict__ clusters,
                                            T* __restrict__ sum_gt0, int* __restrict__ count_gt0,
                                            int nrows, int n_cls) {
  int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= nrows) return;
  int start_idx = indptr[cell];
  int stop_idx = indptr[cell + 1];
  int cluster = clusters[cell];
  for (int gene = start_idx; gene < stop_idx; gene++) {
    T value = data[gene];
    int gene_number = index[gene];
    if (value > (T)0) {
      atomicAdd(&sum_gt0[gene_number * n_cls + cluster], value);
      atomicAdd(&count_gt0[gene_number * n_cls + cluster], 1);
    }
  }
}

template <typename T>
__global__ void mean_dense_kernel(const T* __restrict__ data, const int* __restrict__ clusters,
                                  T* __restrict__ g_cluster, int num_rows, int num_cols,
                                  int n_cls) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= num_rows || j >= num_cols) return;
  atomicAdd(&g_cluster[j * n_cls + clusters[i]], data[i * num_cols + j]);
}

template <typename T>
__global__ void mean_sparse_kernel(const int* __restrict__ indptr, const int* __restrict__ index,
                                   const T* __restrict__ data, const int* __restrict__ clusters,
                                   T* __restrict__ sum_gt0, int nrows, int n_cls) {
  int cell = blockDim.x * blockIdx.x + threadIdx.x;
  if (cell >= nrows) return;
  int start_idx = indptr[cell];
  int stop_idx = indptr[cell + 1];
  int cluster = clusters[cell];
  for (int gene = start_idx; gene < stop_idx; gene++) {
    T value = data[gene];
    int gene_number = index[gene];
    if (value > (T)0) {
      atomicAdd(&sum_gt0[gene_number * n_cls + cluster], value);
    }
  }
}

template <typename T>
__global__ void elementwise_diff_kernel(T* __restrict__ g_cluster,
                                        const T* __restrict__ total_counts, int num_genes,
                                        int num_clusters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= num_genes || j >= num_clusters) return;
  g_cluster[i * num_clusters + j] = g_cluster[i * num_clusters + j] / total_counts[j];
}

template <typename T>
__global__ void interaction_kernel(const int* __restrict__ interactions,
                                   const int* __restrict__ interaction_clusters,
                                   const T* __restrict__ mean, T* __restrict__ res,
                                   const bool* __restrict__ mask, const T* __restrict__ g,
                                   int n_iter, int n_inter_clust, int n_cls) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_iter || j >= n_inter_clust) return;
  int rec = interactions[i * 2];
  int lig = interactions[i * 2 + 1];
  int c1 = interaction_clusters[j * 2];
  int c2 = interaction_clusters[j * 2 + 1];
  T m1 = mean[rec * n_cls + c1];
  T m2 = mean[lig * n_cls + c2];
  if (!isnan(res[i * n_inter_clust + j])) {
    if (m1 > (T)0 && m2 > (T)0) {
      if (mask[rec * n_cls + c1] && mask[lig * n_cls + c2]) {
        T g_sum = g[rec * n_cls + c1] + g[lig * n_cls + c2];
        res[i * n_inter_clust + j] += (g_sum > (m1 + m2));
      } else {
        res[i * n_inter_clust + j] = nan("");
      }
    } else {
      res[i * n_inter_clust + j] = nan("");
    }
  }
}

template <typename T>
__global__ void res_mean_kernel(const int* __restrict__ interactions,
                                const int* __restrict__ interaction_clusters,
                                const T* __restrict__ mean, T* __restrict__ res_mean, int n_inter,
                                int n_inter_clust, int n_cls) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n_inter || j >= n_inter_clust) return;
  int rec = interactions[i * 2];
  int lig = interactions[i * 2 + 1];
  int c1 = interaction_clusters[j * 2];
  int c2 = interaction_clusters[j * 2 + 1];
  T m1 = mean[rec * n_cls + c1];
  T m2 = mean[lig * n_cls + c2];
  if (m1 > (T)0 && m2 > (T)0) {
    res_mean[i * n_inter_clust + j] = (m1 + m2) / (T)2;
  }
}
