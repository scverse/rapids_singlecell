#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_ligrec.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_sum_count_dense(const T* data, const int* clusters, T* sum, int* count,
                                          int rows, int cols, int ncls, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
  sum_and_count_dense_kernel<T>
      <<<grid, block, 0, stream>>>(data, clusters, sum, count, rows, cols, ncls);
}

template <typename T>
static inline void launch_sum_count_sparse(const int* indptr, const int* index, const T* data,
                                           const int* clusters, T* sum, int* count, int rows,
                                           int ncls, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((rows + block.x - 1) / block.x);
  sum_and_count_sparse_kernel<T>
      <<<grid, block, 0, stream>>>(indptr, index, data, clusters, sum, count, rows, ncls);
}

template <typename T>
static inline void launch_mean_dense(const T* data, const int* clusters, T* g, int rows, int cols,
                                     int ncls, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
  mean_dense_kernel<T><<<grid, block, 0, stream>>>(data, clusters, g, rows, cols, ncls);
}

template <typename T>
static inline void launch_mean_sparse(const int* indptr, const int* index, const T* data,
                                      const int* clusters, T* g, int rows, int ncls,
                                      cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((rows + block.x - 1) / block.x);
  mean_sparse_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, clusters, g, rows, ncls);
}

template <typename T>
static inline void launch_elementwise_diff(T* g, const T* total_counts, int n_genes, int n_clusters,
                                           cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((n_genes + block.x - 1) / block.x, (n_clusters + block.y - 1) / block.y);
  elementwise_diff_kernel<T><<<grid, block, 0, stream>>>(g, total_counts, n_genes, n_clusters);
}

template <typename T>
static inline void launch_interaction(const int* interactions, const int* interaction_clusters,
                                      const T* mean, T* res, const bool* mask, const T* g,
                                      int n_iter, int n_inter_clust, int ncls,
                                      cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((n_iter + block.x - 1) / block.x, (n_inter_clust + block.y - 1) / block.y);
  interaction_kernel<T><<<grid, block>>>(interactions, interaction_clusters, mean, res, mask, g,
                                         n_iter, n_inter_clust, ncls);
}

template <typename T>
static inline void launch_res_mean(const int* interactions, const int* interaction_clusters,
                                   const T* mean, T* res_mean, int n_inter, int n_inter_clust,
                                   int ncls, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((n_inter + block.x - 1) / block.x, (n_inter_clust + block.y - 1) / block.y);
  res_mean_kernel<T><<<grid, block, 0, stream>>>(interactions, interaction_clusters, mean, res_mean,
                                                 n_inter, n_inter_clust, ncls);
}

NB_MODULE(_ligrec_cuda, m) {
  // sum_count_dense - float32
  m.def(
      "sum_count_dense",
      [](cuda_array_c<const float> data, cuda_array_c<const int> clusters, cuda_array_c<float> sum,
         cuda_array_c<int> count, int rows, int cols, int ncls, std::uintptr_t stream) {
        launch_sum_count_dense<float>(data.data(), clusters.data(), sum.data(), count.data(), rows,
                                      cols, ncls, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "clusters"_a, "sum"_a, "count"_a, "rows"_a, "cols"_a, "ncls"_a,
      "stream"_a = 0);

  // sum_count_dense - float64
  m.def(
      "sum_count_dense",
      [](cuda_array_c<const double> data, cuda_array_c<const int> clusters,
         cuda_array_c<double> sum, cuda_array_c<int> count, int rows, int cols, int ncls,
         std::uintptr_t stream) {
        launch_sum_count_dense<double>(data.data(), clusters.data(), sum.data(), count.data(), rows,
                                       cols, ncls, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "clusters"_a, "sum"_a, "count"_a, "rows"_a, "cols"_a, "ncls"_a,
      "stream"_a = 0);

  // sum_count_sparse - float32
  m.def(
      "sum_count_sparse",
      [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
         cuda_array_c<const float> data, cuda_array_c<const int> clusters, cuda_array_c<float> sum,
         cuda_array_c<int> count, int rows, int ncls, std::uintptr_t stream) {
        launch_sum_count_sparse<float>(indptr.data(), index.data(), data.data(), clusters.data(),
                                       sum.data(), count.data(), rows, ncls, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "clusters"_a, "sum"_a, "count"_a, "rows"_a,
      "ncls"_a, "stream"_a = 0);

  // sum_count_sparse - float64
  m.def(
      "sum_count_sparse",
      [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
         cuda_array_c<const double> data, cuda_array_c<const int> clusters,
         cuda_array_c<double> sum, cuda_array_c<int> count, int rows, int ncls,
         std::uintptr_t stream) {
        launch_sum_count_sparse<double>(indptr.data(), index.data(), data.data(), clusters.data(),
                                        sum.data(), count.data(), rows, ncls, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "clusters"_a, "sum"_a, "count"_a, "rows"_a,
      "ncls"_a, "stream"_a = 0);

  // mean_dense - float32
  m.def(
      "mean_dense",
      [](cuda_array_c<const float> data, cuda_array_c<const int> clusters, cuda_array_c<float> g,
         int rows, int cols, int ncls, std::uintptr_t stream) {
        launch_mean_dense<float>(data.data(), clusters.data(), g.data(), rows, cols, ncls,
                                 (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "clusters"_a, "g"_a, "rows"_a, "cols"_a, "ncls"_a, "stream"_a = 0);

  // mean_dense - float64
  m.def(
      "mean_dense",
      [](cuda_array_c<const double> data, cuda_array_c<const int> clusters, cuda_array_c<double> g,
         int rows, int cols, int ncls, std::uintptr_t stream) {
        launch_mean_dense<double>(data.data(), clusters.data(), g.data(), rows, cols, ncls,
                                  (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "clusters"_a, "g"_a, "rows"_a, "cols"_a, "ncls"_a, "stream"_a = 0);

  // mean_sparse - float32
  m.def(
      "mean_sparse",
      [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
         cuda_array_c<const float> data, cuda_array_c<const int> clusters, cuda_array_c<float> g,
         int rows, int ncls, std::uintptr_t stream) {
        launch_mean_sparse<float>(indptr.data(), index.data(), data.data(), clusters.data(),
                                  g.data(), rows, ncls, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "clusters"_a, "g"_a, "rows"_a, "ncls"_a,
      "stream"_a = 0);

  // mean_sparse - float64
  m.def(
      "mean_sparse",
      [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
         cuda_array_c<const double> data, cuda_array_c<const int> clusters, cuda_array_c<double> g,
         int rows, int ncls, std::uintptr_t stream) {
        launch_mean_sparse<double>(indptr.data(), index.data(), data.data(), clusters.data(),
                                   g.data(), rows, ncls, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "clusters"_a, "g"_a, "rows"_a, "ncls"_a,
      "stream"_a = 0);

  // elementwise_diff - float32
  m.def(
      "elementwise_diff",
      [](cuda_array_c<float> g, cuda_array_c<const float> total_counts, int n_genes, int n_clusters,
         std::uintptr_t stream) {
        launch_elementwise_diff<float>(g.data(), total_counts.data(), n_genes, n_clusters,
                                       (cudaStream_t)stream);
      },
      "g"_a, nb::kw_only(), "total_counts"_a, "n_genes"_a, "n_clusters"_a, "stream"_a = 0);

  // elementwise_diff - float64
  m.def(
      "elementwise_diff",
      [](cuda_array_c<double> g, cuda_array_c<const double> total_counts, int n_genes,
         int n_clusters, std::uintptr_t stream) {
        launch_elementwise_diff<double>(g.data(), total_counts.data(), n_genes, n_clusters,
                                        (cudaStream_t)stream);
      },
      "g"_a, nb::kw_only(), "total_counts"_a, "n_genes"_a, "n_clusters"_a, "stream"_a = 0);

  // interaction - float32
  m.def(
      "interaction",
      [](cuda_array_c<const int> interactions, cuda_array_c<const int> interaction_clusters,
         cuda_array_c<const float> mean, cuda_array_c<float> res, cuda_array_c<const bool> mask,
         cuda_array_c<const float> g, int n_iter, int n_inter_clust, int ncls,
         std::uintptr_t stream) {
        launch_interaction<float>(interactions.data(), interaction_clusters.data(), mean.data(),
                                  res.data(), mask.data(), g.data(), n_iter, n_inter_clust, ncls,
                                  (cudaStream_t)stream);
      },
      "interactions"_a, nb::kw_only(), "interaction_clusters"_a, "mean"_a, "res"_a, "mask"_a, "g"_a,
      "n_iter"_a, "n_inter_clust"_a, "ncls"_a, "stream"_a = 0);

  // interaction - float64
  m.def(
      "interaction",
      [](cuda_array_c<const int> interactions, cuda_array_c<const int> interaction_clusters,
         cuda_array_c<const double> mean, cuda_array_c<double> res, cuda_array_c<const bool> mask,
         cuda_array_c<const double> g, int n_iter, int n_inter_clust, int ncls,
         std::uintptr_t stream) {
        launch_interaction<double>(interactions.data(), interaction_clusters.data(), mean.data(),
                                   res.data(), mask.data(), g.data(), n_iter, n_inter_clust, ncls,
                                   (cudaStream_t)stream);
      },
      "interactions"_a, nb::kw_only(), "interaction_clusters"_a, "mean"_a, "res"_a, "mask"_a, "g"_a,
      "n_iter"_a, "n_inter_clust"_a, "ncls"_a, "stream"_a = 0);

  // res_mean - float32
  m.def(
      "res_mean",
      [](cuda_array_c<const int> interactions, cuda_array_c<const int> interaction_clusters,
         cuda_array_c<const float> mean, cuda_array_c<float> res_mean, int n_inter,
         int n_inter_clust, int ncls, std::uintptr_t stream) {
        launch_res_mean<float>(interactions.data(), interaction_clusters.data(), mean.data(),
                               res_mean.data(), n_inter, n_inter_clust, ncls, (cudaStream_t)stream);
      },
      "interactions"_a, nb::kw_only(), "interaction_clusters"_a, "mean"_a, "res_mean"_a,
      "n_inter"_a, "n_inter_clust"_a, "ncls"_a, "stream"_a = 0);

  // res_mean - float64
  m.def(
      "res_mean",
      [](cuda_array_c<const int> interactions, cuda_array_c<const int> interaction_clusters,
         cuda_array_c<const double> mean, cuda_array_c<double> res_mean, int n_inter,
         int n_inter_clust, int ncls, std::uintptr_t stream) {
        launch_res_mean<double>(interactions.data(), interaction_clusters.data(), mean.data(),
                                res_mean.data(), n_inter, n_inter_clust, ncls,
                                (cudaStream_t)stream);
      },
      "interactions"_a, nb::kw_only(), "interaction_clusters"_a, "mean"_a, "res_mean"_a,
      "n_inter"_a, "n_inter_clust"_a, "ncls"_a, "stream"_a = 0);
}
