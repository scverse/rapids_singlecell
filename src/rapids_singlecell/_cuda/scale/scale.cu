#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_scale.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE_SPARSE = 64;
constexpr int BLOCK_SIZE_DENSE_TILE = 32;

template <typename T, typename IdxT>
static inline void launch_csc_scale_diff(const IdxT* indptr, T* data,
                                         const T* std, int ncols,
                                         cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_SPARSE);
    dim3 grid(ncols);
    csc_scale_diff_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indptr, data, std, ncols);
    CUDA_CHECK_LAST_ERROR(csc_scale_diff_kernel);
}

template <typename T, typename IdxT>
static inline void launch_csr_scale_diff(const IdxT* indptr,
                                         const IdxT* indices, T* data,
                                         const T* std, const int* mask,
                                         T clipper, int nrows,
                                         cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_SPARSE);
    dim3 grid(nrows);
    csr_scale_diff_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, std, mask, clipper, nrows);
    CUDA_CHECK_LAST_ERROR(csr_scale_diff_kernel);
}

template <typename T>
static inline void launch_dense_scale_center_diff(T* data, const T* mean,
                                                  const T* std, const int* mask,
                                                  T clipper, long long nrows,
                                                  long long ncols,
                                                  cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_DENSE_TILE, BLOCK_SIZE_DENSE_TILE);
    dim3 grid((unsigned)((nrows + block.x - 1) / block.x),
              (unsigned)((ncols + block.y - 1) / block.y));
    dense_scale_center_diff_kernel<T><<<grid, block, 0, stream>>>(
        data, mean, std, mask, clipper, nrows, ncols);
    CUDA_CHECK_LAST_ERROR(dense_scale_center_diff_kernel);
}

template <typename T>
static inline void launch_dense_scale_diff(T* data, const T* std,
                                           const int* mask, T clipper,
                                           long long nrows, long long ncols,
                                           cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_DENSE_TILE, BLOCK_SIZE_DENSE_TILE);
    dim3 grid((unsigned)((nrows + block.x - 1) / block.x),
              (unsigned)((ncols + block.y - 1) / block.y));
    dense_scale_diff_kernel<T>
        <<<grid, block, 0, stream>>>(data, std, mask, clipper, nrows, ncols);
    CUDA_CHECK_LAST_ERROR(dense_scale_diff_kernel);
}

template <typename IdxT, typename T, typename Device>
void def_csc_scale_diff(nb::module_& m) {
    m.def(
        "csc_scale_diff",
        [](gpu_array_c<const IdxT, Device> indptr, gpu_array_c<T, Device> data,
           gpu_array_c<const T, Device> std, int ncols, std::uintptr_t stream) {
            launch_csc_scale_diff<T, IdxT>(indptr.data(), data.data(),
                                           std.data(), ncols,
                                           (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, "std"_a, nb::kw_only(), "ncols"_a,
        "stream"_a = 0);
}

template <typename IdxT, typename T, typename Device>
void def_csr_scale_diff(nb::module_& m) {
    m.def(
        "csr_scale_diff",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices, gpu_array_c<T, Device> data,
           gpu_array_c<const T, Device> std,
           gpu_array_c<const int, Device> mask, T clipper, int nrows,
           std::uintptr_t stream) {
            launch_csr_scale_diff<T, IdxT>(
                indptr.data(), indices.data(), data.data(), std.data(),
                mask.data(), clipper, nrows, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "std"_a, "mask"_a, nb::kw_only(),
        "clipper"_a, "nrows"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // csc_scale_diff - int32 indices
    def_csc_scale_diff<int, float, Device>(m);
    def_csc_scale_diff<int, double, Device>(m);
    // csc_scale_diff - int64 indices
    def_csc_scale_diff<long long, float, Device>(m);
    def_csc_scale_diff<long long, double, Device>(m);

    // csr_scale_diff - int32 indices
    def_csr_scale_diff<int, float, Device>(m);
    def_csr_scale_diff<int, double, Device>(m);
    // csr_scale_diff - int64 indices
    def_csr_scale_diff<long long, float, Device>(m);
    def_csr_scale_diff<long long, double, Device>(m);

    // dense_scale_center_diff - float32
    m.def(
        "dense_scale_center_diff",
        [](gpu_array_c<float, Device> data,
           gpu_array_c<const float, Device> mean,
           gpu_array_c<const float, Device> std,
           gpu_array_c<const int, Device> mask, float clipper, long long nrows,
           long long ncols, std::uintptr_t stream) {
            launch_dense_scale_center_diff<float>(
                data.data(), mean.data(), std.data(), mask.data(), clipper,
                nrows, ncols, (cudaStream_t)stream);
        },
        "data"_a, "mean"_a, "std"_a, "mask"_a, nb::kw_only(), "clipper"_a,
        "nrows"_a, "ncols"_a, "stream"_a = 0);

    // dense_scale_center_diff - float64
    m.def(
        "dense_scale_center_diff",
        [](gpu_array_c<double, Device> data,
           gpu_array_c<const double, Device> mean,
           gpu_array_c<const double, Device> std,
           gpu_array_c<const int, Device> mask, double clipper, long long nrows,
           long long ncols, std::uintptr_t stream) {
            launch_dense_scale_center_diff<double>(
                data.data(), mean.data(), std.data(), mask.data(), clipper,
                nrows, ncols, (cudaStream_t)stream);
        },
        "data"_a, "mean"_a, "std"_a, "mask"_a, nb::kw_only(), "clipper"_a,
        "nrows"_a, "ncols"_a, "stream"_a = 0);

    // dense_scale_diff - float32
    m.def(
        "dense_scale_diff",
        [](gpu_array_c<float, Device> data,
           gpu_array_c<const float, Device> std,
           gpu_array_c<const int, Device> mask, float clipper, long long nrows,
           long long ncols, std::uintptr_t stream) {
            launch_dense_scale_diff<float>(data.data(), std.data(), mask.data(),
                                           clipper, nrows, ncols,
                                           (cudaStream_t)stream);
        },
        "data"_a, "std"_a, "mask"_a, nb::kw_only(), "clipper"_a, "nrows"_a,
        "ncols"_a, "stream"_a = 0);

    // dense_scale_diff - float64
    m.def(
        "dense_scale_diff",
        [](gpu_array_c<double, Device> data,
           gpu_array_c<const double, Device> std,
           gpu_array_c<const int, Device> mask, double clipper, long long nrows,
           long long ncols, std::uintptr_t stream) {
            launch_dense_scale_diff<double>(data.data(), std.data(),
                                            mask.data(), clipper, nrows, ncols,
                                            (cudaStream_t)stream);
        },
        "data"_a, "std"_a, "mask"_a, nb::kw_only(), "clipper"_a, "nrows"_a,
        "ncols"_a, "stream"_a = 0);
}

NB_MODULE(_scale_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
