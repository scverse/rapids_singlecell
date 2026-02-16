#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_normalize.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_normalize(T* X, long long rows, long long cols, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(rows);
  normalize_kernel<T><<<grid, block, 0, stream>>>(X, rows, cols);
}

NB_MODULE(_harmony_normalize_cuda, m) {
  // normalize - float32
  m.def(
      "normalize",
      [](cuda_array<float> X, long long rows, long long cols, std::uintptr_t stream) {
        launch_normalize<float>(X.data(), rows, cols, (cudaStream_t)stream);
      },
      "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);

  // normalize - float64
  m.def(
      "normalize",
      [](cuda_array<double> X, long long rows, long long cols, std::uintptr_t stream) {
        launch_normalize<double>(X.data(), rows, cols, (cudaStream_t)stream);
      },
      "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);
}
