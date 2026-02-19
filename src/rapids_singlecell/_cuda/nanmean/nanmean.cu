#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_nanmean.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_nan_mean_minor(const int* index, const T* data,
                                         double* means, int* nans,
                                         const bool* mask, int nnz,
                                         cudaStream_t stream) {
    dim3 block(32);
    dim3 grid((nnz + block.x - 1) / block.x);
    nan_mean_minor_kernel<T>
        <<<grid, block, 0, stream>>>(index, data, means, nans, mask, nnz);
}

template <typename T>
static inline void launch_nan_mean_major(const int* indptr, const int* index,
                                         const T* data, double* means,
                                         int* nans, const bool* mask, int major,
                                         int minor, cudaStream_t stream) {
    dim3 block(64);
    dim3 grid(major);
    nan_mean_major_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, means, nans, mask, major, minor);
}

NB_MODULE(_nanmean_cuda, m) {
    // nan_mean_minor - float32
    m.def(
        "nan_mean_minor",
        [](cuda_array_c<const int> index, cuda_array_c<const float> data,
           cuda_array_c<double> means, cuda_array_c<int> nans,
           cuda_array_c<const bool> mask, int nnz, std::uintptr_t stream) {
            launch_nan_mean_minor<float>(index.data(), data.data(),
                                         means.data(), nans.data(), mask.data(),
                                         nnz, (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a, "mask"_a,
        "nnz"_a, "stream"_a = 0);

    // nan_mean_minor - float64
    m.def(
        "nan_mean_minor",
        [](cuda_array_c<const int> index, cuda_array_c<const double> data,
           cuda_array_c<double> means, cuda_array_c<int> nans,
           cuda_array_c<const bool> mask, int nnz, std::uintptr_t stream) {
            launch_nan_mean_minor<double>(
                index.data(), data.data(), means.data(), nans.data(),
                mask.data(), nnz, (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a, "mask"_a,
        "nnz"_a, "stream"_a = 0);

    // nan_mean_major - float32
    m.def(
        "nan_mean_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
           cuda_array_c<const float> data, cuda_array_c<double> means,
           cuda_array_c<int> nans, cuda_array_c<const bool> mask, int major,
           int minor, std::uintptr_t stream) {
            launch_nan_mean_major<float>(
                indptr.data(), index.data(), data.data(), means.data(),
                nans.data(), mask.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a,
        "mask"_a, "major"_a, "minor"_a, "stream"_a = 0);

    // nan_mean_major - float64
    m.def(
        "nan_mean_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
           cuda_array_c<const double> data, cuda_array_c<double> means,
           cuda_array_c<int> nans, cuda_array_c<const bool> mask, int major,
           int minor, std::uintptr_t stream) {
            launch_nan_mean_major<double>(
                indptr.data(), index.data(), data.data(), means.data(),
                nans.data(), mask.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a,
        "mask"_a, "major"_a, "minor"_a, "stream"_a = 0);
}
