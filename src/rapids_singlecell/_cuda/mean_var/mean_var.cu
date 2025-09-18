#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_mv.cuh"

namespace nb = nanobind;
using nb::handle;
using namespace nb::literals;

template <typename T>
static inline void launch_mean_var_major(std::uintptr_t indptr_ptr, std::uintptr_t indices_ptr,
                                         std::uintptr_t data_ptr, std::uintptr_t means_ptr,
                                         std::uintptr_t vars_ptr, int major, int minor,
                                         cudaStream_t stream) {
  dim3 block(64);
  dim3 grid(major);
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  const int* indices = reinterpret_cast<const int*>(indices_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  double* means = reinterpret_cast<double*>(means_ptr);
  double* vars = reinterpret_cast<double*>(vars_ptr);
  mean_var_major_kernel<T>
      <<<grid, block, 0, stream>>>(indptr, indices, data, means, vars, major, minor);
}

template <typename T>
static inline void launch_mean_var_minor(std::uintptr_t indices_ptr, std::uintptr_t data_ptr,
                                         std::uintptr_t means_ptr, std::uintptr_t vars_ptr, int nnz,
                                         cudaStream_t stream) {
  int block = 256;
  int grid = (nnz + block - 1) / block;
  const int* indices = reinterpret_cast<const int*>(indices_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  double* means = reinterpret_cast<double*>(means_ptr);
  double* vars = reinterpret_cast<double*>(vars_ptr);
  mean_var_minor_kernel<T><<<grid, block, 0, stream>>>(indices, data, means, vars, nnz);
}

template <typename T>
void mean_var_major_api(std::uintptr_t indptr, std::uintptr_t indices, std::uintptr_t data,
                        std::uintptr_t means, std::uintptr_t vars, int major, int minor,
                        cudaStream_t stream) {
  launch_mean_var_major<T>(indptr, indices, data, means, vars, major, minor, stream);
}

template <typename T>
void mean_var_minor_api(std::uintptr_t indices, std::uintptr_t data, std::uintptr_t means,
                        std::uintptr_t vars, int nnz, cudaStream_t stream) {
  launch_mean_var_minor<T>(indices, data, means, vars, nnz, stream);
}

NB_MODULE(_mean_var_cuda, m) {
  m.def(
      "mean_var_major",
      [](std::uintptr_t indptr, std::uintptr_t indices, std::uintptr_t data, std::uintptr_t means,
         std::uintptr_t vars, int major, int minor, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          mean_var_major_api<float>(indptr, indices, data, means, vars, major, minor,
                                    (cudaStream_t)stream);
        } else if (itemsize == 8) {
          mean_var_major_api<double>(indptr, indices, data, means, vars, major, minor,
                                     (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize for mean_var_major (expected 4 or 8)");
        }
      },
      "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, "major"_a, "minor"_a, "itemsize"_a,
      "stream"_a = 0);
  m.def(
      "mean_var_minor",
      [](std::uintptr_t indices, std::uintptr_t data, std::uintptr_t means, std::uintptr_t vars,
         int nnz, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          mean_var_minor_api<float>(indices, data, means, vars, nnz, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          mean_var_minor_api<double>(indices, data, means, vars, nnz, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize for mean_var_minor (expected 4 or 8)");
        }
      },
      "indices"_a, "data"_a, "means"_a, "vars"_a, "nnz"_a, "itemsize"_a, "stream"_a = 0);
}
