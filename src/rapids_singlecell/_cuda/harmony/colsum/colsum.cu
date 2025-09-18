#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_colsum.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
static inline void launch_colsum(std::uintptr_t A, std::uintptr_t out, std::size_t rows,
                                 std::size_t cols, cudaStream_t stream) {
  int threads = 32;
  int blocks = (int)cols;
  colsum_kernel<T><<<blocks, threads, 0, stream>>>(reinterpret_cast<const T*>(A),
                                                   reinterpret_cast<T*>(out), rows, cols);
}

template <typename T>
static inline void launch_colsum_atomic(std::uintptr_t A, std::uintptr_t out, std::size_t rows,
                                        std::size_t cols, cudaStream_t stream) {
  int tile_rows = (rows + 31) / 32;
  int tile_cols = (cols + 31) / 32;
  int blocks = tile_rows * tile_cols;
  dim3 threads(32, 32);
  colsum_atomic_kernel<T><<<blocks, threads, 0, stream>>>(reinterpret_cast<const T*>(A),
                                                          reinterpret_cast<T*>(out), rows, cols);
}

NB_MODULE(_harmony_colsum_cuda, m) {
  m.def(
      "colsum",
      [](std::uintptr_t A, std::uintptr_t out, std::size_t rows, std::size_t cols, int dtype_code,
         std::uintptr_t stream) {
        // dtype_code: 0=float32, 1=float64, 2=int32; Back-compat: 4->float32, 8->float64
        if (dtype_code == 0 || dtype_code == 4) {
          launch_colsum<float>(A, out, rows, cols, (cudaStream_t)stream);
        } else if (dtype_code == 1 || dtype_code == 8) {
          launch_colsum<double>(A, out, rows, cols, (cudaStream_t)stream);
        } else if (dtype_code == 2) {
          launch_colsum<int>(A, out, rows, cols, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported dtype_code (expected 0/1/2 or 4/8)");
        }
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "dtype_code"_a, "stream"_a = 0);

  m.def(
      "colsum_atomic",
      [](std::uintptr_t A, std::uintptr_t out, std::size_t rows, std::size_t cols, int dtype_code,
         std::uintptr_t stream) {
        if (dtype_code == 0 || dtype_code == 4) {
          launch_colsum_atomic<float>(A, out, rows, cols, (cudaStream_t)stream);
        } else if (dtype_code == 1 || dtype_code == 8) {
          launch_colsum_atomic<double>(A, out, rows, cols, (cudaStream_t)stream);
        } else if (dtype_code == 2) {
          launch_colsum_atomic<int>(A, out, rows, cols, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported dtype_code (expected 0/1/2 or 4/8)");
        }
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "dtype_code"_a, "stream"_a = 0);
}
