#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_autocorr.cuh"

namespace nb = nanobind;

static inline void launch_morans_dense(std::uintptr_t data_centered, std::uintptr_t adj_row_ptr,
                                       std::uintptr_t adj_col_ind, std::uintptr_t adj_data,
                                       std::uintptr_t num, int n_samples, int n_features) {
  dim3 block(8, 8);
  dim3 grid((n_features + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);
  morans_I_num_dense_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(data_centered), reinterpret_cast<const int*>(adj_row_ptr),
      reinterpret_cast<const int*>(adj_col_ind), reinterpret_cast<const float*>(adj_data),
      reinterpret_cast<float*>(num), n_samples, n_features);
}

static inline void launch_morans_sparse(std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind,
                                        std::uintptr_t adj_data, std::uintptr_t data_row_ptr,
                                        std::uintptr_t data_col_ind, std::uintptr_t data_values,
                                        int n_samples, int n_features, std::uintptr_t mean_array,
                                        std::uintptr_t num) {
  dim3 block(1024);
  dim3 grid(n_samples);
  morans_I_num_sparse_kernel<<<grid, block>>>(
      reinterpret_cast<const int*>(adj_row_ptr), reinterpret_cast<const int*>(adj_col_ind),
      reinterpret_cast<const float*>(adj_data), reinterpret_cast<const int*>(data_row_ptr),
      reinterpret_cast<const int*>(data_col_ind), reinterpret_cast<const float*>(data_values),
      n_samples, n_features, reinterpret_cast<const float*>(mean_array),
      reinterpret_cast<float*>(num));
}

static inline void launch_gearys_dense(std::uintptr_t data, std::uintptr_t adj_row_ptr,
                                       std::uintptr_t adj_col_ind, std::uintptr_t adj_data,
                                       std::uintptr_t num, int n_samples, int n_features) {
  dim3 block(8, 8);
  dim3 grid((n_features + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);
  gearys_C_num_dense_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(data), reinterpret_cast<const int*>(adj_row_ptr),
      reinterpret_cast<const int*>(adj_col_ind), reinterpret_cast<const float*>(adj_data),
      reinterpret_cast<float*>(num), n_samples, n_features);
}

static inline void launch_gearys_sparse(std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind,
                                        std::uintptr_t adj_data, std::uintptr_t data_row_ptr,
                                        std::uintptr_t data_col_ind, std::uintptr_t data_values,
                                        int n_samples, int n_features, std::uintptr_t num) {
  dim3 block(1024);
  dim3 grid(n_samples);
  gearys_C_num_sparse_kernel<<<grid, block>>>(
      reinterpret_cast<const int*>(adj_row_ptr), reinterpret_cast<const int*>(adj_col_ind),
      reinterpret_cast<const float*>(adj_data), reinterpret_cast<const int*>(data_row_ptr),
      reinterpret_cast<const int*>(data_col_ind), reinterpret_cast<const float*>(data_values),
      n_samples, n_features, reinterpret_cast<float*>(num));
}

static inline void launch_pre_den_sparse(std::uintptr_t data_col_ind, std::uintptr_t data_values,
                                         int nnz, std::uintptr_t mean_array, std::uintptr_t den,
                                         std::uintptr_t counter) {
  dim3 block(32);
  dim3 grid((nnz + block.x - 1) / block.x);
  pre_den_sparse_kernel<<<grid, block>>>(
      reinterpret_cast<const int*>(data_col_ind), reinterpret_cast<const float*>(data_values), nnz,
      reinterpret_cast<const float*>(mean_array), reinterpret_cast<float*>(den),
      reinterpret_cast<int*>(counter));
}

NB_MODULE(_autocorr_cuda, m) {
  m.def("morans_dense", &launch_morans_dense);
  m.def("morans_sparse", &launch_morans_sparse);
  m.def("gearys_dense", &launch_gearys_dense);
  m.def("gearys_sparse", &launch_gearys_sparse);
  m.def("pre_den_sparse", &launch_pre_den_sparse);
}
