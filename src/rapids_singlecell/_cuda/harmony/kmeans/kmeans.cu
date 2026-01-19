#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_kmeans.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_kmeans_err(const T* r, const T* dot, std::size_t n, T* out,
                                     cudaStream_t stream) {
  int threads = 256;
  int blocks = min((int)((n + threads - 1) / threads), (int)(8 * 128));
  kmeans_err_kernel<T><<<blocks, threads, 0, stream>>>(r, dot, n, out);
}

NB_MODULE(_harmony_kmeans_cuda, m) {
  // kmeans_err - float32
  m.def(
      "kmeans_err",
      [](cuda_array<const float> r, cuda_array<const float> dot, std::size_t n,
         cuda_array<float> out, std::uintptr_t stream) {
        launch_kmeans_err<float>(r.data(), dot.data(), n, out.data(), (cudaStream_t)stream);
      },
      "r"_a, nb::kw_only(), "dot"_a, "n"_a, "out"_a, "stream"_a = 0);

  // kmeans_err - float64
  m.def(
      "kmeans_err",
      [](cuda_array<const double> r, cuda_array<const double> dot, std::size_t n,
         cuda_array<double> out, std::uintptr_t stream) {
        launch_kmeans_err<double>(r.data(), dot.data(), n, out.data(), (cudaStream_t)stream);
      },
      "r"_a, nb::kw_only(), "dot"_a, "n"_a, "out"_a, "stream"_a = 0);
}
