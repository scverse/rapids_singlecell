#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_mv.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_mean_var_major(const int* indptr, const int* indices,
                                         const T* data, double* means,
                                         double* vars, int major, int minor,
                                         cudaStream_t stream) {
    dim3 block(64);
    dim3 grid(major);
    mean_var_major_kernel<T><<<grid, block, 0, stream>>>(
        indptr, indices, data, means, vars, major, minor);
}

template <typename T>
static inline void launch_mean_var_minor(const int* indices, const T* data,
                                         double* means, double* vars, int nnz,
                                         cudaStream_t stream) {
    int block = 256;
    int grid = (nnz + block - 1) / block;
    mean_var_minor_kernel<T>
        <<<grid, block, 0, stream>>>(indices, data, means, vars, nnz);
}

NB_MODULE(_mean_var_cuda, m) {
    // Float32 major
    m.def(
        "mean_var_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<const float> data, cuda_array_c<double> means,
           cuda_array_c<double> vars, int major, int minor,
           std::uintptr_t stream) {
            launch_mean_var_major<float>(indptr.data(), indices.data(),
                                         data.data(), means.data(), vars.data(),
                                         major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);

    // Float64 major
    m.def(
        "mean_var_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<const double> data, cuda_array_c<double> means,
           cuda_array_c<double> vars, int major, int minor,
           std::uintptr_t stream) {
            launch_mean_var_major<double>(
                indptr.data(), indices.data(), data.data(), means.data(),
                vars.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);

    // Float32 minor
    m.def(
        "mean_var_minor",
        [](cuda_array_c<const int> indices, cuda_array_c<const float> data,
           cuda_array_c<double> means, cuda_array_c<double> vars, int nnz,
           std::uintptr_t stream) {
            launch_mean_var_minor<float>(indices.data(), data.data(),
                                         means.data(), vars.data(), nnz,
                                         (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);

    // Float64 minor
    m.def(
        "mean_var_minor",
        [](cuda_array_c<const int> indices, cuda_array_c<const double> data,
           cuda_array_c<double> means, cuda_array_c<double> vars, int nnz,
           std::uintptr_t stream) {
            launch_mean_var_minor<double>(indices.data(), data.data(),
                                          means.data(), vars.data(), nnz,
                                          (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);
}
