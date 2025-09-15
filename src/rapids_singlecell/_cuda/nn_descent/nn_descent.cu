#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_dist.cuh"

namespace nb = nanobind;

static inline void launch_sqeuclidean(std::uintptr_t data, std::uintptr_t out, std::uintptr_t pairs,
                                      long long n_samples, long long n_features,
                                      long long n_neighbors) {
  dim3 block(32);
  dim3 grid((unsigned)((n_samples + block.x - 1) / block.x));
  compute_distances_sqeuclidean_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(data), reinterpret_cast<float*>(out),
      reinterpret_cast<const unsigned int*>(pairs), n_samples, n_features, n_neighbors);
}

static inline void launch_cosine(std::uintptr_t data, std::uintptr_t out, std::uintptr_t pairs,
                                 long long n_samples, long long n_features, long long n_neighbors) {
  dim3 block(32);
  dim3 grid((unsigned)((n_samples + block.x - 1) / block.x));
  compute_distances_cosine_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(data), reinterpret_cast<float*>(out),
      reinterpret_cast<const unsigned int*>(pairs), n_samples, n_features, n_neighbors);
}

static inline void launch_inner(std::uintptr_t data, std::uintptr_t out, std::uintptr_t pairs,
                                long long n_samples, long long n_features, long long n_neighbors) {
  dim3 block(32);
  dim3 grid((unsigned)((n_samples + block.x - 1) / block.x));
  compute_distances_inner_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(data), reinterpret_cast<float*>(out),
      reinterpret_cast<const unsigned int*>(pairs), n_samples, n_features, n_neighbors);
}

NB_MODULE(_nn_descent_cuda, m) {
  m.def("sqeuclidean", &launch_sqeuclidean);
  m.def("cosine", &launch_cosine);
  m.def("inner", &launch_inner);
}
