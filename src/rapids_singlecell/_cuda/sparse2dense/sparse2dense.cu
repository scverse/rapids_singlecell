#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_sparse2dense(std::uintptr_t indptr_ptr, std::uintptr_t index_ptr,
                                       std::uintptr_t data_ptr, std::uintptr_t out_ptr,
                                       long long major, long long minor, int c_switch,
                                       int max_nnz) {
  dim3 block(32, 32);
  dim3 grid((unsigned)((major + block.x - 1) / block.x),
            (unsigned)((max_nnz + block.y - 1) / block.y));
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  const int* index = reinterpret_cast<const int*>(index_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  T* out = reinterpret_cast<T*>(out_ptr);
  sparse2dense_kernel<T><<<grid, block>>>(indptr, index, data, out, major, minor, c_switch);
}

NB_MODULE(_sparse2dense_cuda, m) {
  m.def("sparse2dense",
        [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, std::uintptr_t out,
           long long major, long long minor, int c_switch, int max_nnz, int itemsize) {
          if (itemsize == 4) {
            launch_sparse2dense<float>(indptr, index, data, out, major, minor, c_switch, max_nnz);
          } else if (itemsize == 8) {
            launch_sparse2dense<double>(indptr, index, data, out, major, minor, c_switch, max_nnz);
          } else {
            throw nb::value_error("Unsupported itemsize for sparse2dense (expected 4 or 8)");
          }
        });
}
