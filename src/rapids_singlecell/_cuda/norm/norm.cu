#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_norm.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
static inline void launch_dense_row_scale(std::uintptr_t data_ptr, int nrows, int ncols,
                                          T target_sum, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((nrows + block.x - 1) / block.x);
  T* data = reinterpret_cast<T*>(data_ptr);
  dense_row_scale_kernel<T><<<grid, block, 0, stream>>>(data, nrows, ncols, target_sum);
}

template <typename T>
static inline void launch_csr_row_scale(std::uintptr_t indptr_ptr, std::uintptr_t data_ptr,
                                        int nrows, T target_sum, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((nrows + block.x - 1) / block.x);
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  T* data = reinterpret_cast<T*>(data_ptr);
  csr_row_scale_kernel<T><<<grid, block, 0, stream>>>(indptr, data, nrows, target_sum);
}

template <typename T>
static inline void launch_csr_sum_major(std::uintptr_t indptr_ptr, std::uintptr_t data_ptr,
                                        std::uintptr_t sums_ptr, int major, cudaStream_t stream) {
  dim3 block(64);
  dim3 grid(major);
  std::size_t smem = static_cast<std::size_t>(block.x) * sizeof(T);
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  T* sums = reinterpret_cast<T*>(sums_ptr);
  csr_sum_major_kernel<T><<<grid, block, smem, stream>>>(indptr, data, sums, major);
}

NB_MODULE(_norm_cuda, m) {
  m.def(
      "mul_dense",
      [](std::uintptr_t data, int nrows, int ncols, double target_sum, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_dense_row_scale<float>(data, nrows, ncols, (float)target_sum,
                                        (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_dense_row_scale<double>(data, nrows, ncols, target_sum, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "data"_a, "nrows"_a, "ncols"_a, "target_sum"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "mul_csr",
      [](std::uintptr_t indptr, std::uintptr_t data, int nrows, double target_sum, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_csr_row_scale<float>(indptr, data, nrows, (float)target_sum, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_csr_row_scale<double>(indptr, data, nrows, target_sum, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "indptr"_a, "data"_a, "nrows"_a, "target_sum"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "sum_major",
      [](std::uintptr_t indptr, std::uintptr_t data, std::uintptr_t sums, int major, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_csr_sum_major<float>(indptr, data, sums, major, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_csr_sum_major<double>(indptr, data, sums, major, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "indptr"_a, "data"_a, "sums"_a, "major"_a, "itemsize"_a, "stream"_a = 0);
}
