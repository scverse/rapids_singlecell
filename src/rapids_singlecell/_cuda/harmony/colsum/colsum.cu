#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_colsum.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_colsum(const T* A, T* out, std::size_t rows, std::size_t cols,
                                 cudaStream_t stream) {
  int threads = 32;
  int blocks = (int)cols;
  colsum_kernel<T><<<blocks, threads, 0, stream>>>(A, out, rows, cols);
}

template <typename T>
static inline void launch_colsum_atomic(const T* A, T* out, std::size_t rows, std::size_t cols,
                                        cudaStream_t stream) {
  int tile_rows = (rows + 31) / 32;
  int tile_cols = (cols + 31) / 32;
  int blocks = tile_rows * tile_cols;
  dim3 threads(32, 32);
  colsum_atomic_kernel<T><<<blocks, threads, 0, stream>>>(A, out, rows, cols);
}

NB_MODULE(_harmony_colsum_cuda, m) {
  // colsum - float32
  m.def(
      "colsum",
      [](cuda_array<const float> A, cuda_array<float> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum<float>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum - float64
  m.def(
      "colsum",
      [](cuda_array<const double> A, cuda_array<double> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum<double>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum - int32
  m.def(
      "colsum",
      [](cuda_array<const int> A, cuda_array<int> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum<int>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum_atomic - float32
  m.def(
      "colsum_atomic",
      [](cuda_array<const float> A, cuda_array<float> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum_atomic<float>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum_atomic - float64
  m.def(
      "colsum_atomic",
      [](cuda_array<const double> A, cuda_array<double> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum_atomic<double>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum_atomic - int32
  m.def(
      "colsum_atomic",
      [](cuda_array<const int> A, cuda_array<int> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum_atomic<int>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);
}
