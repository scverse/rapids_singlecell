#include <algorithm>
#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_colsum.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_colsum(const T* A, T* out, std::size_t rows, std::size_t cols,
                                 cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int max_blocks = prop.multiProcessorCount * 8;

  // Scale thread count with rows, capped at 1024, minimum 32
  int threads = std::min(1024, std::max(32, (int)((rows + 31) / 32) * 32));
  int blocks = std::min((int)cols, max_blocks);
  colsum_kernel<T><<<blocks, threads, 0, stream>>>(A, out, rows, cols);
}

template <typename T>
static inline void launch_colsum_atomic(const T* A, T* out, std::size_t rows, std::size_t cols,
                                        cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  int col_tiles = (int)((cols + 31) / 32);
  int target_row_tiles = std::max(1, n_sm * 4 / std::max(1, col_tiles));
  std::size_t rows_per_tile =
      std::max((std::size_t)32, (rows + target_row_tiles - 1) / target_row_tiles);

  int row_tiles = (int)((rows + rows_per_tile - 1) / rows_per_tile);
  dim3 grid(col_tiles, row_tiles);
  dim3 threads(32, 32);
  colsum_atomic_kernel<T><<<grid, threads, 0, stream>>>(A, out, rows, cols, rows_per_tile);
}

NB_MODULE(_harmony_colsum_cuda, m) {
  // colsum - float32
  m.def(
      "colsum",
      [](cuda_array_c<const float> A, cuda_array_c<float> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum<float>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum - float64
  m.def(
      "colsum",
      [](cuda_array_c<const double> A, cuda_array_c<double> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum<double>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum - int32
  m.def(
      "colsum",
      [](cuda_array_c<const int> A, cuda_array_c<int> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum<int>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum_atomic - float32
  m.def(
      "colsum_atomic",
      [](cuda_array_c<const float> A, cuda_array_c<float> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum_atomic<float>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum_atomic - float64
  m.def(
      "colsum_atomic",
      [](cuda_array_c<const double> A, cuda_array_c<double> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum_atomic<double>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);

  // colsum_atomic - int32
  m.def(
      "colsum_atomic",
      [](cuda_array_c<const int> A, cuda_array_c<int> out, std::size_t rows, std::size_t cols,
         std::uintptr_t stream) {
        launch_colsum_atomic<int>(A.data(), out.data(), rows, cols, (cudaStream_t)stream);
      },
      "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);
}
