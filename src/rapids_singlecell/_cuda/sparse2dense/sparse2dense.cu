#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_s2d.cuh"

namespace nb = nanobind;

template <typename T, bool C_ORDER>
static inline void launch_typed(const int* indptr, const int* index, const T* data, T* out,
                                long long major, long long minor, int max_nnz, dim3 grid,
                                dim3 block) {
  sparse2dense_kernel<T, C_ORDER><<<grid, block>>>(indptr, index, data, out, major, minor);
}

template <typename T>
static inline void launch_sparse2dense(std::uintptr_t indptr_ptr, std::uintptr_t index_ptr,
                                       std::uintptr_t data_ptr, std::uintptr_t out_ptr,
                                       long long major, long long minor,
                                       bool c_switch,  // 1 = C (row-major), 0 = F (col-major)
                                       int max_nnz) {
  // Threads: 32x32 (1024) as you had; adjust if register pressure is high.
  dim3 block(32, 32);
  dim3 grid((unsigned)((major + block.x - 1) / block.x),
            (unsigned)((max_nnz + block.y - 1) / block.y));

  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  const int* index = reinterpret_cast<const int*>(index_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  T* out = reinterpret_cast<T*>(out_ptr);

  if (c_switch == true) {
    launch_typed<T, /*C_ORDER=*/true>(indptr, index, data, out, major, minor, max_nnz, grid, block);
  } else {
    launch_typed<T, /*C_ORDER=*/false>(indptr, index, data, out, major, minor, max_nnz, grid,
                                       block);
  }
}

NB_MODULE(_sparse2dense_cuda, m) {
  m.def("sparse2dense",
        [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, std::uintptr_t out,
           long long major, long long minor, bool c_switch, int max_nnz, int itemsize) {
          if (itemsize == 4) {
            launch_sparse2dense<float>(indptr, index, data, out, major, minor, c_switch, max_nnz);
          } else if (itemsize == 8) {
            launch_sparse2dense<double>(indptr, index, data, out, major, minor, c_switch, max_nnz);
          } else {
            throw nb::value_error("Unsupported itemsize for sparse2dense (expected 4 or 8)");
          }
        });
}
