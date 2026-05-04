#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_autocorr.cuh"

using namespace nb::literals;

constexpr int DENSE_BLOCK_DIM = 8;
constexpr int SPARSE_BLOCK_SIZE = 1024;
constexpr int ELEMENTWISE_BLOCK_SIZE = 32;

template <typename T>
static inline void launch_morans_dense(const T* data_centered,
                                       const int* adj_row_ptr,
                                       const int* adj_col_ind,
                                       const T* adj_data, T* num, int n_samples,
                                       int n_features, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_features + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_samples + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    morans_I_num_dense_kernel<<<grid, block, 0, stream>>>(
        data_centered, adj_row_ptr, adj_col_ind, adj_data, num, n_samples,
        n_features);
    CUDA_CHECK_LAST_ERROR(morans_I_num_dense_kernel);
}

template <typename T>
static inline void launch_morans_sparse(
    const int* adj_row_ptr, const int* adj_col_ind, const T* adj_data,
    const int* data_row_ptr, const int* data_col_ind, const T* data_values,
    int n_samples, int n_features, const T* mean_array, T* num,
    cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid(n_samples);
    morans_I_num_sparse_kernel<<<grid, block, 0, stream>>>(
        adj_row_ptr, adj_col_ind, adj_data, data_row_ptr, data_col_ind,
        data_values, n_samples, n_features, mean_array, num);
    CUDA_CHECK_LAST_ERROR(morans_I_num_sparse_kernel);
}

template <typename T>
static inline void launch_gearys_dense(const T* data, const int* adj_row_ptr,
                                       const int* adj_col_ind,
                                       const T* adj_data, T* num, int n_samples,
                                       int n_features, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_features + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_samples + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    gearys_C_num_dense_kernel<<<grid, block, 0, stream>>>(
        data, adj_row_ptr, adj_col_ind, adj_data, num, n_samples, n_features);
    CUDA_CHECK_LAST_ERROR(gearys_C_num_dense_kernel);
}

template <typename T>
static inline void launch_gearys_sparse(
    const int* adj_row_ptr, const int* adj_col_ind, const T* adj_data,
    const int* data_row_ptr, const int* data_col_ind, const T* data_values,
    int n_samples, int n_features, T* num, cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid(n_samples);
    gearys_C_num_sparse_kernel<<<grid, block, 0, stream>>>(
        adj_row_ptr, adj_col_ind, adj_data, data_row_ptr, data_col_ind,
        data_values, n_samples, n_features, num);
    CUDA_CHECK_LAST_ERROR(gearys_C_num_sparse_kernel);
}

template <typename T, typename IdxT>
static inline void launch_pre_den_sparse(const IdxT* data_col_ind,
                                         const T* data_values, long long nnz,
                                         const T* mean_array, T* den,
                                         int* counter, cudaStream_t stream) {
    dim3 block(ELEMENTWISE_BLOCK_SIZE);
    long long grid_size =
        (nnz + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    dim3 grid(grid_size);
    pre_den_sparse_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        data_col_ind, data_values, nnz, mean_array, den, counter);
    CUDA_CHECK_LAST_ERROR(pre_den_sparse_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // morans_dense - float32
    m.def(
        "morans_dense",
        [](gpu_array_c<const float, Device> data_centered,
           gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const float, Device> adj_data,
           gpu_array_c<float, Device> num, int n_samples, int n_features,
           std::uintptr_t stream) {
            launch_morans_dense(data_centered.data(), adj_row_ptr.data(),
                                adj_col_ind.data(), adj_data.data(), num.data(),
                                n_samples, n_features, (cudaStream_t)stream);
        },
        "data_centered"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a,
        "adj_data"_a, "num"_a, "n_samples"_a, "n_features"_a, "stream"_a = 0);
    // morans_dense - float64
    m.def(
        "morans_dense",
        [](gpu_array_c<const double, Device> data_centered,
           gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const double, Device> adj_data,
           gpu_array_c<double, Device> num, int n_samples, int n_features,
           std::uintptr_t stream) {
            launch_morans_dense(data_centered.data(), adj_row_ptr.data(),
                                adj_col_ind.data(), adj_data.data(), num.data(),
                                n_samples, n_features, (cudaStream_t)stream);
        },
        "data_centered"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a,
        "adj_data"_a, "num"_a, "n_samples"_a, "n_features"_a, "stream"_a = 0);

    // morans_sparse - float32
    m.def(
        "morans_sparse",
        [](gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const float, Device> adj_data,
           gpu_array_c<const int, Device> data_row_ptr,
           gpu_array_c<const int, Device> data_col_ind,
           gpu_array_c<const float, Device> data_values, int n_samples,
           int n_features, gpu_array_c<const float, Device> mean_array,
           gpu_array_c<float, Device> num, std::uintptr_t stream) {
            launch_morans_sparse(adj_row_ptr.data(), adj_col_ind.data(),
                                 adj_data.data(), data_row_ptr.data(),
                                 data_col_ind.data(), data_values.data(),
                                 n_samples, n_features, mean_array.data(),
                                 num.data(), (cudaStream_t)stream);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "n_samples"_a,
        "n_features"_a, "mean_array"_a, "num"_a, "stream"_a = 0);
    // morans_sparse - float64
    m.def(
        "morans_sparse",
        [](gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const double, Device> adj_data,
           gpu_array_c<const int, Device> data_row_ptr,
           gpu_array_c<const int, Device> data_col_ind,
           gpu_array_c<const double, Device> data_values, int n_samples,
           int n_features, gpu_array_c<const double, Device> mean_array,
           gpu_array_c<double, Device> num, std::uintptr_t stream) {
            launch_morans_sparse(adj_row_ptr.data(), adj_col_ind.data(),
                                 adj_data.data(), data_row_ptr.data(),
                                 data_col_ind.data(), data_values.data(),
                                 n_samples, n_features, mean_array.data(),
                                 num.data(), (cudaStream_t)stream);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "n_samples"_a,
        "n_features"_a, "mean_array"_a, "num"_a, "stream"_a = 0);

    // gearys_dense - float32
    m.def(
        "gearys_dense",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const float, Device> adj_data,
           gpu_array_c<float, Device> num, int n_samples, int n_features,
           std::uintptr_t stream) {
            launch_gearys_dense(data.data(), adj_row_ptr.data(),
                                adj_col_ind.data(), adj_data.data(), num.data(),
                                n_samples, n_features, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a,
        "num"_a, "n_samples"_a, "n_features"_a, "stream"_a = 0);
    // gearys_dense - float64
    m.def(
        "gearys_dense",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const double, Device> adj_data,
           gpu_array_c<double, Device> num, int n_samples, int n_features,
           std::uintptr_t stream) {
            launch_gearys_dense(data.data(), adj_row_ptr.data(),
                                adj_col_ind.data(), adj_data.data(), num.data(),
                                n_samples, n_features, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a,
        "num"_a, "n_samples"_a, "n_features"_a, "stream"_a = 0);

    // gearys_sparse - float32
    m.def(
        "gearys_sparse",
        [](gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const float, Device> adj_data,
           gpu_array_c<const int, Device> data_row_ptr,
           gpu_array_c<const int, Device> data_col_ind,
           gpu_array_c<const float, Device> data_values, int n_samples,
           int n_features, gpu_array_c<float, Device> num,
           std::uintptr_t stream) {
            launch_gearys_sparse(
                adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                n_samples, n_features, num.data(), (cudaStream_t)stream);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "n_samples"_a,
        "n_features"_a, "num"_a, "stream"_a = 0);
    // gearys_sparse - float64
    m.def(
        "gearys_sparse",
        [](gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const double, Device> adj_data,
           gpu_array_c<const int, Device> data_row_ptr,
           gpu_array_c<const int, Device> data_col_ind,
           gpu_array_c<const double, Device> data_values, int n_samples,
           int n_features, gpu_array_c<double, Device> num,
           std::uintptr_t stream) {
            launch_gearys_sparse(
                adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                n_samples, n_features, num.data(), (cudaStream_t)stream);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "n_samples"_a,
        "n_features"_a, "num"_a, "stream"_a = 0);

    // pre_den_sparse - float32, int32
    m.def(
        "pre_den_sparse",
        [](gpu_array_c<const int, Device> data_col_ind,
           gpu_array_c<const float, Device> data_values, long long nnz,
           gpu_array_c<const float, Device> mean_array,
           gpu_array_c<float, Device> den, gpu_array_c<int, Device> counter,
           std::uintptr_t stream) {
            launch_pre_den_sparse<float, int>(
                data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                den.data(), counter.data(), (cudaStream_t)stream);
        },
        "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a,
        "mean_array"_a, "den"_a, "counter"_a, "stream"_a = 0);
    // pre_den_sparse - float64, int32
    m.def(
        "pre_den_sparse",
        [](gpu_array_c<const int, Device> data_col_ind,
           gpu_array_c<const double, Device> data_values, long long nnz,
           gpu_array_c<const double, Device> mean_array,
           gpu_array_c<double, Device> den, gpu_array_c<int, Device> counter,
           std::uintptr_t stream) {
            launch_pre_den_sparse<double, int>(
                data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                den.data(), counter.data(), (cudaStream_t)stream);
        },
        "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a,
        "mean_array"_a, "den"_a, "counter"_a, "stream"_a = 0);
    // pre_den_sparse - float32, int64
    m.def(
        "pre_den_sparse",
        [](gpu_array_c<const long long, Device> data_col_ind,
           gpu_array_c<const float, Device> data_values, long long nnz,
           gpu_array_c<const float, Device> mean_array,
           gpu_array_c<float, Device> den, gpu_array_c<int, Device> counter,
           std::uintptr_t stream) {
            launch_pre_den_sparse<float, long long>(
                data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                den.data(), counter.data(), (cudaStream_t)stream);
        },
        "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a,
        "mean_array"_a, "den"_a, "counter"_a, "stream"_a = 0);
    // pre_den_sparse - float64, int64
    m.def(
        "pre_den_sparse",
        [](gpu_array_c<const long long, Device> data_col_ind,
           gpu_array_c<const double, Device> data_values, long long nnz,
           gpu_array_c<const double, Device> mean_array,
           gpu_array_c<double, Device> den, gpu_array_c<int, Device> counter,
           std::uintptr_t stream) {
            launch_pre_den_sparse<double, long long>(
                data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                den.data(), counter.data(), (cudaStream_t)stream);
        },
        "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a,
        "mean_array"_a, "den"_a, "counter"_a, "stream"_a = 0);
}

NB_MODULE(_autocorr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
