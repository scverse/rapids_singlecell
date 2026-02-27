#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_scale.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_csc_scale_diff(const int* indptr, T* data,
                                         const T* std, int ncols,
                                         cudaStream_t stream) {
    dim3 block(64);
    dim3 grid(ncols);
    csc_scale_diff_kernel<T>
        <<<grid, block, 0, stream>>>(indptr, data, std, ncols);
}

template <typename T>
static inline void launch_csr_scale_diff(const int* indptr, const int* indices,
                                         T* data, const T* std, const int* mask,
                                         T clipper, int nrows,
                                         cudaStream_t stream) {
    dim3 block(64);
    dim3 grid(nrows);
    csr_scale_diff_kernel<T><<<grid, block, 0, stream>>>(
        indptr, indices, data, std, mask, clipper, nrows);
}

template <typename T>
static inline void launch_dense_scale_center_diff(T* data, const T* mean,
                                                  const T* std, const int* mask,
                                                  T clipper, long long nrows,
                                                  long long ncols,
                                                  cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((unsigned)((nrows + block.x - 1) / block.x),
              (unsigned)((ncols + block.y - 1) / block.y));
    dense_scale_center_diff_kernel<T><<<grid, block, 0, stream>>>(
        data, mean, std, mask, clipper, nrows, ncols);
}

template <typename T>
static inline void launch_dense_scale_diff(T* data, const T* std,
                                           const int* mask, T clipper,
                                           long long nrows, long long ncols,
                                           cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((unsigned)((nrows + block.x - 1) / block.x),
              (unsigned)((ncols + block.y - 1) / block.y));
    dense_scale_diff_kernel<T>
        <<<grid, block, 0, stream>>>(data, std, mask, clipper, nrows, ncols);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // csc_scale_diff - float32
    m.def(
        "csc_scale_diff",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<float, Device> data,
           gpu_array_c<const float, Device> std, int ncols,
           std::uintptr_t stream) {
            launch_csc_scale_diff<float>(indptr.data(), data.data(), std.data(),
                                         ncols, (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, "std"_a, nb::kw_only(), "ncols"_a,
        "stream"_a = 0);

    // csc_scale_diff - float64
    m.def(
        "csc_scale_diff",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<double, Device> data,
           gpu_array_c<const double, Device> std, int ncols,
           std::uintptr_t stream) {
            launch_csc_scale_diff<double>(indptr.data(), data.data(),
                                          std.data(), ncols,
                                          (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, "std"_a, nb::kw_only(), "ncols"_a,
        "stream"_a = 0);

    // csr_scale_diff - float32
    m.def(
        "csr_scale_diff",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<float, Device> data,
           gpu_array_c<const float, Device> std,
           gpu_array_c<const int, Device> mask, float clipper, int nrows,
           std::uintptr_t stream) {
            launch_csr_scale_diff<float>(indptr.data(), indices.data(),
                                         data.data(), std.data(), mask.data(),
                                         clipper, nrows, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "std"_a, "mask"_a, nb::kw_only(),
        "clipper"_a, "nrows"_a, "stream"_a = 0);

    // csr_scale_diff - float64
    m.def(
        "csr_scale_diff",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<double, Device> data,
           gpu_array_c<const double, Device> std,
           gpu_array_c<const int, Device> mask, double clipper, int nrows,
           std::uintptr_t stream) {
            launch_csr_scale_diff<double>(indptr.data(), indices.data(),
                                          data.data(), std.data(), mask.data(),
                                          clipper, nrows, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "std"_a, "mask"_a, nb::kw_only(),
        "clipper"_a, "nrows"_a, "stream"_a = 0);

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
