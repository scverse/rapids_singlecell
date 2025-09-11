#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_csc_scale_diff(std::uintptr_t indptr, std::uintptr_t data,
                                         std::uintptr_t std, int ncols) {
  dim3 block(64);
  dim3 grid(ncols);
  csc_scale_diff_kernel<T><<<grid, block>>>(reinterpret_cast<const int*>(indptr),
                                            reinterpret_cast<T*>(data),
                                            reinterpret_cast<const T*>(std), ncols);
}

template <typename T>
static inline void launch_csr_scale_diff(std::uintptr_t indptr, std::uintptr_t indices,
                                         std::uintptr_t data, std::uintptr_t std,
                                         std::uintptr_t mask, T clipper, int nrows) {
  dim3 block(64);
  dim3 grid(nrows);
  csr_scale_diff_kernel<T>
      <<<grid, block>>>(reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(indices),
                        reinterpret_cast<T*>(data), reinterpret_cast<const T*>(std),
                        reinterpret_cast<const int*>(mask), clipper, nrows);
}

template <typename T>
static inline void launch_dense_scale_center_diff(std::uintptr_t data, std::uintptr_t mean,
                                                  std::uintptr_t std, std::uintptr_t mask,
                                                  T clipper, long long nrows, long long ncols) {
  dim3 block(32, 32);
  dim3 grid((unsigned)((nrows + block.x - 1) / block.x),
            (unsigned)((ncols + block.y - 1) / block.y));
  dense_scale_center_diff_kernel<T><<<grid, block>>>(
      reinterpret_cast<T*>(data), reinterpret_cast<const T*>(mean), reinterpret_cast<const T*>(std),
      reinterpret_cast<const int*>(mask), clipper, nrows, ncols);
}

template <typename T>
static inline void launch_dense_scale_diff(std::uintptr_t data, std::uintptr_t std,
                                           std::uintptr_t mask, T clipper, long long nrows,
                                           long long ncols) {
  dim3 block(32, 32);
  dim3 grid((unsigned)((nrows + block.x - 1) / block.x),
            (unsigned)((ncols + block.y - 1) / block.y));
  dense_scale_diff_kernel<T>
      <<<grid, block>>>(reinterpret_cast<T*>(data), reinterpret_cast<const T*>(std),
                        reinterpret_cast<const int*>(mask), clipper, nrows, ncols);
}

NB_MODULE(_scale_cuda, m) {
  m.def("csc_scale_diff", [](std::uintptr_t indptr, std::uintptr_t data, std::uintptr_t std,
                             int ncols, int itemsize) {
    if (itemsize == 4)
      launch_csc_scale_diff<float>(indptr, data, std, ncols);
    else if (itemsize == 8)
      launch_csc_scale_diff<double>(indptr, data, std, ncols);
    else
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
  });
  m.def("csr_scale_diff",
        [](std::uintptr_t indptr, std::uintptr_t indices, std::uintptr_t data, std::uintptr_t std,
           std::uintptr_t mask, double clipper, int nrows, int itemsize) {
          if (itemsize == 4)
            launch_csr_scale_diff<float>(indptr, indices, data, std, mask, (float)clipper, nrows);
          else if (itemsize == 8)
            launch_csr_scale_diff<double>(indptr, indices, data, std, mask, (double)clipper, nrows);
          else
            throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        });
  m.def("dense_scale_center_diff", [](std::uintptr_t data, std::uintptr_t mean, std::uintptr_t std,
                                      std::uintptr_t mask, double clipper, long long nrows,
                                      long long ncols, int itemsize) {
    if (itemsize == 4)
      launch_dense_scale_center_diff<float>(data, mean, std, mask, (float)clipper, nrows, ncols);
    else if (itemsize == 8)
      launch_dense_scale_center_diff<double>(data, mean, std, mask, (double)clipper, nrows, ncols);
    else
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
  });
  m.def("dense_scale_diff", [](std::uintptr_t data, std::uintptr_t std, std::uintptr_t mask,
                               double clipper, long long nrows, long long ncols, int itemsize) {
    if (itemsize == 4)
      launch_dense_scale_diff<float>(data, std, mask, (float)clipper, nrows, ncols);
    else if (itemsize == 8)
      launch_dense_scale_diff<double>(data, std, mask, (double)clipper, nrows, ncols);
    else
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
  });
}
