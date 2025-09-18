#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_ligrec.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
static inline void launch_sum_count_dense(std::uintptr_t data, std::uintptr_t clusters,
                                          std::uintptr_t sum, std::uintptr_t count, int rows,
                                          int cols, int ncls, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
  sum_and_count_dense_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(data), reinterpret_cast<const int*>(clusters),
      reinterpret_cast<T*>(sum), reinterpret_cast<int*>(count), rows, cols, ncls);
}

template <typename T>
static inline void launch_sum_count_sparse(std::uintptr_t indptr, std::uintptr_t index,
                                           std::uintptr_t data, std::uintptr_t clusters,
                                           std::uintptr_t sum, std::uintptr_t count, int rows,
                                           int ncls, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((rows + block.x - 1) / block.x);
  sum_and_count_sparse_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<const int*>(clusters),
      reinterpret_cast<T*>(sum), reinterpret_cast<int*>(count), rows, ncls);
}

template <typename T>
static inline void launch_mean_dense(std::uintptr_t data, std::uintptr_t clusters, std::uintptr_t g,
                                     int rows, int cols, int ncls, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
  mean_dense_kernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(data),
                                                   reinterpret_cast<const int*>(clusters),
                                                   reinterpret_cast<T*>(g), rows, cols, ncls);
}

template <typename T>
static inline void launch_mean_sparse(std::uintptr_t indptr, std::uintptr_t index,
                                      std::uintptr_t data, std::uintptr_t clusters,
                                      std::uintptr_t g, int rows, int ncls, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((rows + block.x - 1) / block.x);
  mean_sparse_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<const int*>(clusters),
      reinterpret_cast<T*>(g), rows, ncls);
}

template <typename T>
static inline void launch_elementwise_diff(std::uintptr_t g, std::uintptr_t total_counts,
                                           int n_genes, int n_clusters, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((n_genes + block.x - 1) / block.x, (n_clusters + block.y - 1) / block.y);
  elementwise_diff_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<T*>(g), reinterpret_cast<const T*>(total_counts), n_genes, n_clusters);
}

template <typename T>
static inline void launch_interaction(std::uintptr_t interactions,
                                      std::uintptr_t interaction_clusters, std::uintptr_t mean,
                                      std::uintptr_t res, std::uintptr_t mask, std::uintptr_t g,
                                      int n_iter, int n_inter_clust, int ncls,
                                      cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((n_iter + block.x - 1) / block.x, (n_inter_clust + block.y - 1) / block.y);
  interaction_kernel<T><<<grid, block>>>(
      reinterpret_cast<const int*>(interactions),
      reinterpret_cast<const int*>(interaction_clusters), reinterpret_cast<const T*>(mean),
      reinterpret_cast<T*>(res), reinterpret_cast<const bool*>(mask), reinterpret_cast<const T*>(g),
      n_iter, n_inter_clust, ncls);
}

template <typename T>
static inline void launch_res_mean(std::uintptr_t interactions, std::uintptr_t interaction_clusters,
                                   std::uintptr_t mean, std::uintptr_t res_mean, int n_inter,
                                   int n_inter_clust, int ncls, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((n_inter + block.x - 1) / block.x, (n_inter_clust + block.y - 1) / block.y);
  res_mean_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(interactions),
      reinterpret_cast<const int*>(interaction_clusters), reinterpret_cast<const T*>(mean),
      reinterpret_cast<T*>(res_mean), n_inter, n_inter_clust, ncls);
}

NB_MODULE(_ligrec_cuda, m) {
  m.def(
      "sum_count_dense",
      [](std::uintptr_t data, std::uintptr_t clusters, std::uintptr_t sum, std::uintptr_t count,
         int rows, int cols, int ncls, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_sum_count_dense<float>(data, clusters, sum, count, rows, cols, ncls,
                                        (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_sum_count_dense<double>(data, clusters, sum, count, rows, cols, ncls,
                                         (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "data"_a, "clusters"_a, "sum"_a, "count"_a, "rows"_a, "cols"_a, "ncls"_a, "itemsize"_a,
      "stream"_a = 0);

  m.def(
      "sum_count_sparse",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, std::uintptr_t clusters,
         std::uintptr_t sum, std::uintptr_t count, int rows, int ncls, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_sum_count_sparse<float>(indptr, index, data, clusters, sum, count, rows, ncls,
                                         (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_sum_count_sparse<double>(indptr, index, data, clusters, sum, count, rows, ncls,
                                          (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "indptr"_a, "index"_a, "data"_a, "clusters"_a, "sum"_a, "count"_a, "rows"_a, "ncls"_a,
      "itemsize"_a, "stream"_a = 0);

  m.def(
      "mean_dense",
      [](std::uintptr_t data, std::uintptr_t clusters, std::uintptr_t g, int rows, int cols,
         int ncls, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_mean_dense<float>(data, clusters, g, rows, cols, ncls, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_mean_dense<double>(data, clusters, g, rows, cols, ncls, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "data"_a, "clusters"_a, "g"_a, "rows"_a, "cols"_a, "ncls"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "mean_sparse",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, std::uintptr_t clusters,
         std::uintptr_t g, int rows, int ncls, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_mean_sparse<float>(indptr, index, data, clusters, g, rows, ncls,
                                    (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_mean_sparse<double>(indptr, index, data, clusters, g, rows, ncls,
                                     (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "indptr"_a, "index"_a, "data"_a, "clusters"_a, "g"_a, "rows"_a, "ncls"_a, "itemsize"_a,
      "stream"_a = 0);

  m.def(
      "elementwise_diff",
      [](std::uintptr_t g, std::uintptr_t total_counts, int n_genes, int n_clusters, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_elementwise_diff<float>(g, total_counts, n_genes, n_clusters,
                                         (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_elementwise_diff<double>(g, total_counts, n_genes, n_clusters,
                                          (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "g"_a, "total_counts"_a, "n_genes"_a, "n_clusters"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "interaction",
      [](std::uintptr_t interactions, std::uintptr_t interaction_clusters, std::uintptr_t mean,
         std::uintptr_t res, std::uintptr_t mask, std::uintptr_t g, int n_iter, int n_inter_clust,
         int ncls, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_interaction<float>(interactions, interaction_clusters, mean, res, mask, g, n_iter,
                                    n_inter_clust, ncls, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_interaction<double>(interactions, interaction_clusters, mean, res, mask, g, n_iter,
                                     n_inter_clust, ncls, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "interactions"_a, "interaction_clusters"_a, "mean"_a, "res"_a, "mask"_a, "g"_a, "n_iter"_a,
      "n_inter_clust"_a, "ncls"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "res_mean",
      [](std::uintptr_t interactions, std::uintptr_t interaction_clusters, std::uintptr_t mean,
         std::uintptr_t res_mean, int n_inter, int n_inter_clust, int ncls, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_res_mean<float>(interactions, interaction_clusters, mean, res_mean, n_inter,
                                 n_inter_clust, ncls, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_res_mean<double>(interactions, interaction_clusters, mean, res_mean, n_inter,
                                  n_inter_clust, ncls, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "interactions"_a, "interaction_clusters"_a, "mean"_a, "res_mean"_a, "n_inter"_a,
      "n_inter_clust"_a, "ncls"_a, "itemsize"_a, "stream"_a = 0);
}
