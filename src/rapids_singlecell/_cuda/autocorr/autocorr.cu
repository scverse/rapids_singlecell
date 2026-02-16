#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_autocorr.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_morans_dense(const T* data_centered, const int* adj_row_ptr,
                                       const int* adj_col_ind, const T* adj_data, T* num,
                                       int n_samples, int n_features, cudaStream_t stream) {
  dim3 block(8, 8);
  dim3 grid((n_features + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);
  morans_I_num_dense_kernel<<<grid, block, 0, stream>>>(data_centered, adj_row_ptr, adj_col_ind,
                                                        adj_data, num, n_samples, n_features);
}

template <typename T>
static inline void launch_morans_sparse(const int* adj_row_ptr, const int* adj_col_ind,
                                        const T* adj_data, const int* data_row_ptr,
                                        const int* data_col_ind, const T* data_values,
                                        int n_samples, int n_features, const T* mean_array, T* num,
                                        cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(n_samples);
  morans_I_num_sparse_kernel<<<grid, block, 0, stream>>>(adj_row_ptr, adj_col_ind, adj_data,
                                                         data_row_ptr, data_col_ind, data_values,
                                                         n_samples, n_features, mean_array, num);
}

template <typename T>
static inline void launch_gearys_dense(const T* data, const int* adj_row_ptr,
                                       const int* adj_col_ind, const T* adj_data, T* num,
                                       int n_samples, int n_features, cudaStream_t stream) {
  dim3 block(8, 8);
  dim3 grid((n_features + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);
  gearys_C_num_dense_kernel<<<grid, block, 0, stream>>>(data, adj_row_ptr, adj_col_ind, adj_data,
                                                        num, n_samples, n_features);
}

template <typename T>
static inline void launch_gearys_sparse(const int* adj_row_ptr, const int* adj_col_ind,
                                        const T* adj_data, const int* data_row_ptr,
                                        const int* data_col_ind, const T* data_values,
                                        int n_samples, int n_features, T* num,
                                        cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(n_samples);
  gearys_C_num_sparse_kernel<<<grid, block, 0, stream>>>(adj_row_ptr, adj_col_ind, adj_data,
                                                         data_row_ptr, data_col_ind, data_values,
                                                         n_samples, n_features, num);
}

template <typename T>
static inline void launch_pre_den_sparse(const int* data_col_ind, const T* data_values, int nnz,
                                         const T* mean_array, T* den, int* counter,
                                         cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((nnz + block.x - 1) / block.x);
  pre_den_sparse_kernel<<<grid, block, 0, stream>>>(data_col_ind, data_values, nnz, mean_array, den,
                                                    counter);
}

NB_MODULE(_autocorr_cuda, m) {
  // morans_dense - float32
  m.def(
      "morans_dense",
      [](cuda_array<const float> data_centered, cuda_array<const int> adj_row_ptr,
         cuda_array<const int> adj_col_ind, cuda_array<const float> adj_data, cuda_array<float> num,
         int n_samples, int n_features, std::uintptr_t stream) {
        launch_morans_dense(data_centered.data(), adj_row_ptr.data(), adj_col_ind.data(),
                            adj_data.data(), num.data(), n_samples, n_features,
                            (cudaStream_t)stream);
      },
      "data_centered"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, "num"_a,
      "n_samples"_a, "n_features"_a, "stream"_a = 0);
  // morans_dense - float64
  m.def(
      "morans_dense",
      [](cuda_array<const double> data_centered, cuda_array<const int> adj_row_ptr,
         cuda_array<const int> adj_col_ind, cuda_array<const double> adj_data,
         cuda_array<double> num, int n_samples, int n_features, std::uintptr_t stream) {
        launch_morans_dense(data_centered.data(), adj_row_ptr.data(), adj_col_ind.data(),
                            adj_data.data(), num.data(), n_samples, n_features,
                            (cudaStream_t)stream);
      },
      "data_centered"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, "num"_a,
      "n_samples"_a, "n_features"_a, "stream"_a = 0);

  // morans_sparse - float32
  m.def(
      "morans_sparse",
      [](cuda_array<const int> adj_row_ptr, cuda_array<const int> adj_col_ind,
         cuda_array<const float> adj_data, cuda_array<const int> data_row_ptr,
         cuda_array<const int> data_col_ind, cuda_array<const float> data_values, int n_samples,
         int n_features, cuda_array<const float> mean_array, cuda_array<float> num,
         std::uintptr_t stream) {
        launch_morans_sparse(adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                             data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                             n_samples, n_features, mean_array.data(), num.data(),
                             (cudaStream_t)stream);
      },
      "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(), "data_row_ptr"_a,
      "data_col_ind"_a, "data_values"_a, "n_samples"_a, "n_features"_a, "mean_array"_a, "num"_a,
      "stream"_a = 0);
  // morans_sparse - float64
  m.def(
      "morans_sparse",
      [](cuda_array<const int> adj_row_ptr, cuda_array<const int> adj_col_ind,
         cuda_array<const double> adj_data, cuda_array<const int> data_row_ptr,
         cuda_array<const int> data_col_ind, cuda_array<const double> data_values, int n_samples,
         int n_features, cuda_array<const double> mean_array, cuda_array<double> num,
         std::uintptr_t stream) {
        launch_morans_sparse(adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                             data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                             n_samples, n_features, mean_array.data(), num.data(),
                             (cudaStream_t)stream);
      },
      "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(), "data_row_ptr"_a,
      "data_col_ind"_a, "data_values"_a, "n_samples"_a, "n_features"_a, "mean_array"_a, "num"_a,
      "stream"_a = 0);

  // gearys_dense - float32
  m.def(
      "gearys_dense",
      [](cuda_array<const float> data, cuda_array<const int> adj_row_ptr,
         cuda_array<const int> adj_col_ind, cuda_array<const float> adj_data, cuda_array<float> num,
         int n_samples, int n_features, std::uintptr_t stream) {
        launch_gearys_dense(data.data(), adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                            num.data(), n_samples, n_features, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, "num"_a,
      "n_samples"_a, "n_features"_a, "stream"_a = 0);
  // gearys_dense - float64
  m.def(
      "gearys_dense",
      [](cuda_array<const double> data, cuda_array<const int> adj_row_ptr,
         cuda_array<const int> adj_col_ind, cuda_array<const double> adj_data,
         cuda_array<double> num, int n_samples, int n_features, std::uintptr_t stream) {
        launch_gearys_dense(data.data(), adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                            num.data(), n_samples, n_features, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, "num"_a,
      "n_samples"_a, "n_features"_a, "stream"_a = 0);

  // gearys_sparse - float32
  m.def(
      "gearys_sparse",
      [](cuda_array<const int> adj_row_ptr, cuda_array<const int> adj_col_ind,
         cuda_array<const float> adj_data, cuda_array<const int> data_row_ptr,
         cuda_array<const int> data_col_ind, cuda_array<const float> data_values, int n_samples,
         int n_features, cuda_array<float> num, std::uintptr_t stream) {
        launch_gearys_sparse(adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                             data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                             n_samples, n_features, num.data(), (cudaStream_t)stream);
      },
      "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(), "data_row_ptr"_a,
      "data_col_ind"_a, "data_values"_a, "n_samples"_a, "n_features"_a, "num"_a, "stream"_a = 0);
  // gearys_sparse - float64
  m.def(
      "gearys_sparse",
      [](cuda_array<const int> adj_row_ptr, cuda_array<const int> adj_col_ind,
         cuda_array<const double> adj_data, cuda_array<const int> data_row_ptr,
         cuda_array<const int> data_col_ind, cuda_array<const double> data_values, int n_samples,
         int n_features, cuda_array<double> num, std::uintptr_t stream) {
        launch_gearys_sparse(adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                             data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                             n_samples, n_features, num.data(), (cudaStream_t)stream);
      },
      "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(), "data_row_ptr"_a,
      "data_col_ind"_a, "data_values"_a, "n_samples"_a, "n_features"_a, "num"_a, "stream"_a = 0);

  // pre_den_sparse - float32
  m.def(
      "pre_den_sparse",
      [](cuda_array<const int> data_col_ind, cuda_array<const float> data_values, int nnz,
         cuda_array<const float> mean_array, cuda_array<float> den, cuda_array<int> counter,
         std::uintptr_t stream) {
        launch_pre_den_sparse(data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                              den.data(), counter.data(), (cudaStream_t)stream);
      },
      "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a, "mean_array"_a, "den"_a,
      "counter"_a, "stream"_a = 0);
  // pre_den_sparse - float64
  m.def(
      "pre_den_sparse",
      [](cuda_array<const int> data_col_ind, cuda_array<const double> data_values, int nnz,
         cuda_array<const double> mean_array, cuda_array<double> den, cuda_array<int> counter,
         std::uintptr_t stream) {
        launch_pre_den_sparse(data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                              den.data(), counter.data(), (cudaStream_t)stream);
      },
      "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a, "mean_array"_a, "den"_a,
      "counter"_a, "stream"_a = 0);
}
