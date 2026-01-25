#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_pen.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_pen(T* R, const T* penalty, const int* cats, std::size_t n_rows,
                              std::size_t n_cols, cudaStream_t stream) {
  dim3 block(256);
  std::size_t N = n_rows * n_cols;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  pen_kernel<T><<<grid, block, 0, stream>>>(R, penalty, cats, n_rows, n_cols);
}

NB_MODULE(_harmony_pen_cuda, m) {
  // pen - float32
  m.def(
      "pen",
      [](cuda_array<float> R, cuda_array<const float> penalty, cuda_array<const int> cats,
         std::size_t n_rows, std::size_t n_cols, std::uintptr_t stream) {
        launch_pen<float>(R.data(), penalty.data(), cats.data(), n_rows, n_cols,
                          (cudaStream_t)stream);
      },
      "R"_a, nb::kw_only(), "penalty"_a, "cats"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // pen - float64
  m.def(
      "pen",
      [](cuda_array<double> R, cuda_array<const double> penalty, cuda_array<const int> cats,
         std::size_t n_rows, std::size_t n_cols, std::uintptr_t stream) {
        launch_pen<double>(R.data(), penalty.data(), cats.data(), n_rows, n_cols,
                           (cudaStream_t)stream);
      },
      "R"_a, nb::kw_only(), "penalty"_a, "cats"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}
