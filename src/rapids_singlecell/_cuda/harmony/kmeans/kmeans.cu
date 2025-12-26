#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_kmeans.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
static inline void launch_kmeans_err(std::uintptr_t r, std::uintptr_t dot, std::size_t n,
                                     std::uintptr_t out, cudaStream_t stream) {
  int threads = 256;
  int blocks = min((int)((n + threads - 1) / threads), (int)(8 * 128));
  kmeans_err_kernel<T><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const T*>(r), reinterpret_cast<const T*>(dot), n, reinterpret_cast<T*>(out));
}

NB_MODULE(_harmony_kmeans_cuda, m) {
  m.def(
      "kmeans_err",
      [](std::uintptr_t r, std::uintptr_t dot, std::size_t n, std::uintptr_t out, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_kmeans_err<float>(r, dot, n, out, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_kmeans_err<double>(r, dot, n, out, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "r"_a, nb::kw_only(), "dot"_a, "n"_a, "out"_a, "itemsize"_a, "stream"_a = 0);
}
