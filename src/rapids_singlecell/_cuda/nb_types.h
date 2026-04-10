#pragma once

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

/// Check the last CUDA error after a kernel launch.
/// Call immediately after every <<<...>>> launch to catch configuration errors
/// (invalid grid/block, shared memory overflow, etc.) before they propagate.
inline void cuda_check_last_error(const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(kernel_name) +
                                 " launch failed: " + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK_LAST_ERROR(kernel_name) cuda_check_last_error(#kernel_name)

// GPU array aliases for nanobind bindings, parameterized on device type.
// Bindings are registered for both nb::device::cuda (kDLCUDA = 2) and
// nb::device::cuda_managed (kDLCUDAManaged = 13) so that RMM managed-memory
// allocations are accepted without losing type safety for CPU arrays.

// C-contiguous (row-major)
template <typename T, typename Device>
using gpu_array_c = nb::ndarray<T, Device, nb::c_contig>;

// F-contiguous (column-major)
template <typename T, typename Device>
using gpu_array_f = nb::ndarray<T, Device, nb::f_contig>;

// No contiguity constraint (accepts any order)
template <typename T, typename Device>
using gpu_array = nb::ndarray<T, Device>;

// Parameterized contiguity (for kernels that handle both C and F order)
template <typename T, typename Device, typename Contig>
using gpu_array_contig = nb::ndarray<T, Device, Contig>;

// Host (NumPy) array aliases
template <typename T>
using host_array = nb::ndarray<T, nb::numpy, nb::ndim<1>>;

template <typename T>
using host_array_2d = nb::ndarray<T, nb::numpy>;

// Register bindings for both regular CUDA and managed-memory arrays.
// Usage:
//   template <typename Device>
//   void register_bindings(nb::module_& m) { ... }
//   NB_MODULE(_foo_cuda, m) { REGISTER_GPU_BINDINGS(register_bindings, m); }
#define REGISTER_GPU_BINDINGS(func, module) \
    func<nb::device::cuda>(module);         \
    func<nb::device::cuda_managed>(module)
