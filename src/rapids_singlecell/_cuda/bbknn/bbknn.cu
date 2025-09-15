#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_bbknn.cuh"

namespace nb = nanobind;

static inline void launch_find_top_k_per_row(std::uintptr_t data_ptr, std::uintptr_t indptr_ptr,
                                             int n_rows, int trim, std::uintptr_t vals_ptr) {
  dim3 block(64);
  dim3 grid((n_rows + 64 - 1) / 64);
  std::size_t shared_mem_size =
      static_cast<std::size_t>(64) * static_cast<std::size_t>(trim) * sizeof(float);
  const float* data = reinterpret_cast<const float*>(data_ptr);
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  float* vals = reinterpret_cast<float*>(vals_ptr);
  find_top_k_per_row_kernel<<<grid, block, shared_mem_size>>>(data, indptr, n_rows, trim, vals);
}

static inline void launch_cut_smaller(std::uintptr_t indptr_ptr, std::uintptr_t index_ptr,
                                      std::uintptr_t data_ptr, std::uintptr_t vals_ptr,
                                      int n_rows) {
  dim3 grid(n_rows);
  dim3 block(64);
  int* indptr = reinterpret_cast<int*>(indptr_ptr);
  int* index = reinterpret_cast<int*>(index_ptr);
  float* data = reinterpret_cast<float*>(data_ptr);
  float* vals = reinterpret_cast<float*>(vals_ptr);
  cut_smaller_kernel<<<grid, block>>>(indptr, index, data, vals, n_rows);
}

NB_MODULE(_bbknn_cuda, m) {
  m.def("find_top_k_per_row",
        [](std::uintptr_t data, std::uintptr_t indptr, int n_rows, int trim, std::uintptr_t vals) {
          launch_find_top_k_per_row(data, indptr, n_rows, trim, vals);
        });

  m.def("cut_smaller",
        [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, std::uintptr_t vals,
           int n_rows) { launch_cut_smaller(indptr, index, data, vals, n_rows); });
}
