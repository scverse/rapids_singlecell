#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// CUDA array aliases for nanobind bindings
// C-contiguous (row-major)
template <typename T>
using cuda_array_c = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

// F-contiguous (column-major)
template <typename T>
using cuda_array_f = nb::ndarray<T, nb::device::cuda, nb::f_contig>;

// No contiguity constraint (accepts any order)
template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda>;

// Parameterized contiguity (for kernels that handle both C and F order)
template <typename T, typename Contig>
using cuda_array_contig = nb::ndarray<T, nb::device::cuda, Contig>;
