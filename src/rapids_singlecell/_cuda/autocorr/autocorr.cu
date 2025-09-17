#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_autocorr.cuh"

namespace nb = nanobind;

static inline void launch_morans_dense(std::uintptr_t data_centered, std::uintptr_t adj_row_ptr,
                                       std::uintptr_t adj_col_ind, std::uintptr_t adj_data,
                                       std::uintptr_t num, int n_samples, int n_features,
                                       cudaStream_t stream) {
  dim3 block(8, 8);
  dim3 grid((n_features + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);
  morans_I_num_dense_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const float*>(data_centered), reinterpret_cast<const int*>(adj_row_ptr),
      reinterpret_cast<const int*>(adj_col_ind), reinterpret_cast<const float*>(adj_data),
      reinterpret_cast<float*>(num), n_samples, n_features);
}

static inline void launch_morans_sparse(std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind,
                                        std::uintptr_t adj_data, std::uintptr_t data_row_ptr,
                                        std::uintptr_t data_col_ind, std::uintptr_t data_values,
                                        int n_samples, int n_features, std::uintptr_t mean_array,
                                        std::uintptr_t num, cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(n_samples);
  morans_I_num_sparse_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(adj_row_ptr), reinterpret_cast<const int*>(adj_col_ind),
      reinterpret_cast<const float*>(adj_data), reinterpret_cast<const int*>(data_row_ptr),
      reinterpret_cast<const int*>(data_col_ind), reinterpret_cast<const float*>(data_values),
      n_samples, n_features, reinterpret_cast<const float*>(mean_array),
      reinterpret_cast<float*>(num));
}

static inline void launch_gearys_dense(std::uintptr_t data, std::uintptr_t adj_row_ptr,
                                       std::uintptr_t adj_col_ind, std::uintptr_t adj_data,
                                       std::uintptr_t num, int n_samples, int n_features,
                                       cudaStream_t stream) {
  dim3 block(8, 8);
  dim3 grid((n_features + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);
  gearys_C_num_dense_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const float*>(data), reinterpret_cast<const int*>(adj_row_ptr),
      reinterpret_cast<const int*>(adj_col_ind), reinterpret_cast<const float*>(adj_data),
      reinterpret_cast<float*>(num), n_samples, n_features);
}

static inline void launch_gearys_sparse(std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind,
                                        std::uintptr_t adj_data, std::uintptr_t data_row_ptr,
                                        std::uintptr_t data_col_ind, std::uintptr_t data_values,
                                        int n_samples, int n_features, std::uintptr_t num,
                                        cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(n_samples);
  gearys_C_num_sparse_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(adj_row_ptr), reinterpret_cast<const int*>(adj_col_ind),
      reinterpret_cast<const float*>(adj_data), reinterpret_cast<const int*>(data_row_ptr),
      reinterpret_cast<const int*>(data_col_ind), reinterpret_cast<const float*>(data_values),
      n_samples, n_features, reinterpret_cast<float*>(num));
}

static inline void launch_pre_den_sparse(std::uintptr_t data_col_ind, std::uintptr_t data_values,
                                         int nnz, std::uintptr_t mean_array, std::uintptr_t den,
                                         std::uintptr_t counter, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((nnz + block.x - 1) / block.x);
  pre_den_sparse_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(data_col_ind), reinterpret_cast<const float*>(data_values), nnz,
      reinterpret_cast<const float*>(mean_array), reinterpret_cast<float*>(den),
      reinterpret_cast<int*>(counter));
}

NB_MODULE(_autocorr_cuda, m) {
  m.def(
      "morans_dense",
      [](std::uintptr_t data_centered, std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind,
         std::uintptr_t adj_data, std::uintptr_t num, int n_samples, int n_features,
         std::uintptr_t stream) {
        launch_morans_dense(data_centered, adj_row_ptr, adj_col_ind, adj_data, num, n_samples,
                            n_features, (cudaStream_t)stream);
      },
      nb::arg("data_centered"), nb::arg("adj_row_ptr"), nb::arg("adj_col_ind"), nb::arg("adj_data"),
      nb::arg("num"), nb::arg("n_samples"), nb::arg("n_features"), nb::arg("stream") = 0);
  m.def(
      "morans_sparse",
      [](std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind, std::uintptr_t adj_data,
         std::uintptr_t data_row_ptr, std::uintptr_t data_col_ind, std::uintptr_t data_values,
         int n_samples, int n_features, std::uintptr_t mean_array, std::uintptr_t num,
         std::uintptr_t stream) {
        launch_morans_sparse(adj_row_ptr, adj_col_ind, adj_data, data_row_ptr, data_col_ind,
                             data_values, n_samples, n_features, mean_array, num,
                             (cudaStream_t)stream);
      },
      nb::arg("adj_row_ptr"), nb::arg("adj_col_ind"), nb::arg("adj_data"), nb::arg("data_row_ptr"),
      nb::arg("data_col_ind"), nb::arg("data_values"), nb::arg("n_samples"), nb::arg("n_features"),
      nb::arg("mean_array"), nb::arg("num"), nb::arg("stream") = 0);
  m.def(
      "gearys_dense",
      [](std::uintptr_t data, std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind,
         std::uintptr_t adj_data, std::uintptr_t num, int n_samples, int n_features,
         std::uintptr_t stream) {
        launch_gearys_dense(data, adj_row_ptr, adj_col_ind, adj_data, num, n_samples, n_features,
                            (cudaStream_t)stream);
      },
      nb::arg("data"), nb::arg("adj_row_ptr"), nb::arg("adj_col_ind"), nb::arg("adj_data"),
      nb::arg("num"), nb::arg("n_samples"), nb::arg("n_features"), nb::arg("stream") = 0);
  m.def(
      "gearys_sparse",
      [](std::uintptr_t adj_row_ptr, std::uintptr_t adj_col_ind, std::uintptr_t adj_data,
         std::uintptr_t data_row_ptr, std::uintptr_t data_col_ind, std::uintptr_t data_values,
         int n_samples, int n_features, std::uintptr_t num, std::uintptr_t stream) {
        launch_gearys_sparse(adj_row_ptr, adj_col_ind, adj_data, data_row_ptr, data_col_ind,
                             data_values, n_samples, n_features, num, (cudaStream_t)stream);
      },
      nb::arg("adj_row_ptr"), nb::arg("adj_col_ind"), nb::arg("adj_data"), nb::arg("data_row_ptr"),
      nb::arg("data_col_ind"), nb::arg("data_values"), nb::arg("n_samples"), nb::arg("n_features"),
      nb::arg("num"), nb::arg("stream") = 0);
  m.def(
      "pre_den_sparse",
      [](std::uintptr_t data_col_ind, std::uintptr_t data_values, int nnz,
         std::uintptr_t mean_array, std::uintptr_t den, std::uintptr_t counter,
         std::uintptr_t stream) {
        launch_pre_den_sparse(data_col_ind, data_values, nnz, mean_array, den, counter,
                              (cudaStream_t)stream);
      },
      nb::arg("data_col_ind"), nb::arg("data_values"), nb::arg("nnz"), nb::arg("mean_array"),
      nb::arg("den"), nb::arg("counter"), nb::arg("stream") = 0);
}
