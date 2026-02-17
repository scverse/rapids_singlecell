#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_dist.cuh"

using namespace nb::literals;

static inline void launch_sqeuclidean(const float* data, float* out, const unsigned int* pairs,
                                      long long n_samples, long long n_features,
                                      long long n_neighbors, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((unsigned)((n_samples + block.x - 1) / block.x));
  compute_distances_sqeuclidean_kernel<<<grid, block, 0, stream>>>(data, out, pairs, n_samples,
                                                                   n_features, n_neighbors);
}

static inline void launch_cosine(const float* data, float* out, const unsigned int* pairs,
                                 long long n_samples, long long n_features, long long n_neighbors,
                                 cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((unsigned)((n_samples + block.x - 1) / block.x));
  compute_distances_cosine_kernel<<<grid, block, 0, stream>>>(data, out, pairs, n_samples,
                                                              n_features, n_neighbors);
}

static inline void launch_inner(const float* data, float* out, const unsigned int* pairs,
                                long long n_samples, long long n_features, long long n_neighbors,
                                cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((unsigned)((n_samples + block.x - 1) / block.x));
  compute_distances_inner_kernel<<<grid, block, 0, stream>>>(data, out, pairs, n_samples,
                                                             n_features, n_neighbors);
}

NB_MODULE(_nn_descent_cuda, m) {
  m.def(
      "sqeuclidean",
      [](cuda_array_c<const float> data, cuda_array_c<float> out,
         cuda_array_c<const unsigned int> pairs, long long n_samples, long long n_features,
         long long n_neighbors, std::uintptr_t stream) {
        launch_sqeuclidean(data.data(), out.data(), pairs.data(), n_samples, n_features,
                           n_neighbors, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "out"_a, "pairs"_a, "n_samples"_a, "n_features"_a, "n_neighbors"_a,
      "stream"_a = 0);

  m.def(
      "cosine",
      [](cuda_array_c<const float> data, cuda_array_c<float> out,
         cuda_array_c<const unsigned int> pairs, long long n_samples, long long n_features,
         long long n_neighbors, std::uintptr_t stream) {
        launch_cosine(data.data(), out.data(), pairs.data(), n_samples, n_features, n_neighbors,
                      (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "out"_a, "pairs"_a, "n_samples"_a, "n_features"_a, "n_neighbors"_a,
      "stream"_a = 0);

  m.def(
      "inner",
      [](cuda_array_c<const float> data, cuda_array_c<float> out,
         cuda_array_c<const unsigned int> pairs, long long n_samples, long long n_features,
         long long n_neighbors, std::uintptr_t stream) {
        launch_inner(data.data(), out.data(), pairs.data(), n_samples, n_features, n_neighbors,
                     (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "out"_a, "pairs"_a, "n_samples"_a, "n_features"_a, "n_neighbors"_a,
      "stream"_a = 0);
}
