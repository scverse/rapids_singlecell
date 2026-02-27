#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

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

// Register bindings for both regular CUDA and managed-memory arrays.
// Usage:
//   template <typename Device>
//   void register_bindings(nb::module_& m) { ... }
//   NB_MODULE(_foo_cuda, m) { REGISTER_GPU_BINDINGS(register_bindings, m); }
#define REGISTER_GPU_BINDINGS(func, module) \
    func<nb::device::cuda>(module);         \
    func<nb::device::cuda_managed>(module)
