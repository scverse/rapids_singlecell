#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_mv.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE_MAJOR = 64;
constexpr int BLOCK_SIZE_MINOR = 256;

template <typename T, typename IdxT>
static inline void launch_mean_var_major(const IdxT* indptr,
                                         const IdxT* indices, const T* data,
                                         double* means, double* vars, int major,
                                         int minor, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_MAJOR);
    dim3 grid(major);
    mean_var_major_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, means, vars, major, minor);
    CUDA_CHECK_LAST_ERROR(mean_var_major_kernel);
}

template <typename T, typename IdxT>
static inline void launch_mean_var_minor(const IdxT* indices, const T* data,
                                         double* means, double* vars,
                                         long long nnz, cudaStream_t stream) {
    int block = BLOCK_SIZE_MINOR;
    long long grid = (nnz + block - 1) / block;
    mean_var_minor_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indices, data, means, vars, nnz);
    CUDA_CHECK_LAST_ERROR(mean_var_minor_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // Float32 major, int32
    m.def(
        "mean_var_major",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const float, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           int major, int minor, std::uintptr_t stream) {
            launch_mean_var_major<float>(indptr.data(), indices.data(),
                                         data.data(), means.data(), vars.data(),
                                         major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);

    // Float64 major, int32
    m.def(
        "mean_var_major",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           int major, int minor, std::uintptr_t stream) {
            launch_mean_var_major<double>(
                indptr.data(), indices.data(), data.data(), means.data(),
                vars.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);

    // Float32 major, int64
    m.def(
        "mean_var_major",
        [](gpu_array_c<const long long, Device> indptr,
           gpu_array_c<const long long, Device> indices,
           gpu_array_c<const float, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           int major, int minor, std::uintptr_t stream) {
            launch_mean_var_major<float>(indptr.data(), indices.data(),
                                         data.data(), means.data(), vars.data(),
                                         major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);

    // Float64 major, int64
    m.def(
        "mean_var_major",
        [](gpu_array_c<const long long, Device> indptr,
           gpu_array_c<const long long, Device> indices,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           int major, int minor, std::uintptr_t stream) {
            launch_mean_var_major<double>(
                indptr.data(), indices.data(), data.data(), means.data(),
                vars.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);

    // Float32 minor, int32
    m.def(
        "mean_var_minor",
        [](gpu_array_c<const int, Device> indices,
           gpu_array_c<const float, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           long long nnz, std::uintptr_t stream) {
            launch_mean_var_minor<float>(indices.data(), data.data(),
                                         means.data(), vars.data(), nnz,
                                         (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);

    // Float64 minor, int32
    m.def(
        "mean_var_minor",
        [](gpu_array_c<const int, Device> indices,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           long long nnz, std::uintptr_t stream) {
            launch_mean_var_minor<double>(indices.data(), data.data(),
                                          means.data(), vars.data(), nnz,
                                          (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);

    // Float32 minor, int64
    m.def(
        "mean_var_minor",
        [](gpu_array_c<const long long, Device> indices,
           gpu_array_c<const float, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           long long nnz, std::uintptr_t stream) {
            launch_mean_var_minor<float>(indices.data(), data.data(),
                                         means.data(), vars.data(), nnz,
                                         (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);

    // Float64 minor, int64
    m.def(
        "mean_var_minor",
        [](gpu_array_c<const long long, Device> indices,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> means, gpu_array_c<double, Device> vars,
           long long nnz, std::uintptr_t stream) {
            launch_mean_var_minor<double>(indices.data(), data.data(),
                                          means.data(), vars.data(), nnz,
                                          (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);
}

NB_MODULE(_mean_var_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
