#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_nanmean.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_nan_mean_minor(std::uintptr_t index, std::uintptr_t data,
                                         std::uintptr_t means, std::uintptr_t nans,
                                         std::uintptr_t mask, int nnz, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((nnz + block.x - 1) / block.x);
  nan_mean_minor_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(index), reinterpret_cast<const T*>(data),
      reinterpret_cast<double*>(means), reinterpret_cast<int*>(nans),
      reinterpret_cast<const bool*>(mask), nnz);
}

template <typename T>
static inline void launch_nan_mean_major(std::uintptr_t indptr, std::uintptr_t index,
                                         std::uintptr_t data, std::uintptr_t means,
                                         std::uintptr_t nans, std::uintptr_t mask, int major,
                                         int minor, cudaStream_t stream) {
  dim3 block(64);
  dim3 grid(major);
  nan_mean_major_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<double*>(means),
      reinterpret_cast<int*>(nans), reinterpret_cast<const bool*>(mask), major, minor);
}

NB_MODULE(_nanmean_cuda, m) {
  m.def(
      "nan_mean_minor",
      [](std::uintptr_t index, std::uintptr_t data, std::uintptr_t means, std::uintptr_t nans,
         std::uintptr_t mask, int nnz, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_nan_mean_minor<float>(index, data, means, nans, mask, nnz, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_nan_mean_minor<double>(index, data, means, nans, mask, nnz, (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("index"), nb::arg("data"), nb::arg("means"), nb::arg("nans"), nb::arg("mask"),
      nb::arg("nnz"), nb::arg("itemsize"), nb::arg("stream") = 0);

  m.def(
      "nan_mean_major",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, std::uintptr_t means,
         std::uintptr_t nans, std::uintptr_t mask, int major, int minor, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4)
          launch_nan_mean_major<float>(indptr, index, data, means, nans, mask, major, minor,
                                       (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_nan_mean_major<double>(indptr, index, data, means, nans, mask, major, minor,
                                        (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("indptr"), nb::arg("index"), nb::arg("data"), nb::arg("means"), nb::arg("nans"),
      nb::arg("mask"), nb::arg("major"), nb::arg("minor"), nb::arg("itemsize"),
      nb::arg("stream") = 0);
}
