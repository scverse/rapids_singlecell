#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_spca.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_gram_csr_upper(std::uintptr_t indptr_ptr, std::uintptr_t index_ptr,
                                         std::uintptr_t data_ptr, int nrows, int ncols,
                                         std::uintptr_t out_ptr) {
  dim3 block(128);
  dim3 grid(nrows);
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  const int* index = reinterpret_cast<const int*>(index_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  T* out = reinterpret_cast<T*>(out_ptr);
  gram_csr_upper_kernel<T><<<grid, block>>>(indptr, index, data, nrows, ncols, out);
}

template <typename T>
static inline void launch_copy_upper_to_lower(std::uintptr_t out_ptr, int ncols) {
  dim3 block(32, 32);
  dim3 grid((ncols + block.x - 1) / block.x, (ncols + block.y - 1) / block.y);
  T* out = reinterpret_cast<T*>(out_ptr);
  copy_upper_to_lower_kernel<T><<<grid, block>>>(out, ncols);
}

template <typename T>
static inline void launch_cov_from_gram(std::uintptr_t cov_ptr, std::uintptr_t gram_ptr,
                                        std::uintptr_t meanx_ptr, std::uintptr_t meany_ptr,
                                        int ncols) {
  dim3 block(32, 32);
  dim3 grid((ncols + 31) / 32, (ncols + 31) / 32);
  T* cov = reinterpret_cast<T*>(cov_ptr);
  const T* gram = reinterpret_cast<const T*>(gram_ptr);
  const T* meanx = reinterpret_cast<const T*>(meanx_ptr);
  const T* meany = reinterpret_cast<const T*>(meany_ptr);
  cov_from_gram_kernel<T><<<grid, block>>>(cov, gram, meanx, meany, ncols);
}

static inline void launch_check_zero_genes(std::uintptr_t indices_ptr, std::uintptr_t genes_ptr,
                                           int nnz) {
  dim3 block(32);
  dim3 grid((nnz + block.x - 1) / block.x);
  const int* indices = reinterpret_cast<const int*>(indices_ptr);
  int* genes = reinterpret_cast<int*>(genes_ptr);
  check_zero_genes_kernel<<<grid, block>>>(indices, genes, nnz);
}

NB_MODULE(_spca_cuda, m) {
  m.def("gram_csr_upper", [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
                             int nrows, int ncols, std::uintptr_t out, int itemsize) {
    if (itemsize == 4) {
      launch_gram_csr_upper<float>(indptr, index, data, nrows, ncols, out);
    } else if (itemsize == 8) {
      launch_gram_csr_upper<double>(indptr, index, data, nrows, ncols, out);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });

  m.def("copy_upper_to_lower", [](std::uintptr_t out, int ncols, int itemsize) {
    if (itemsize == 4) {
      launch_copy_upper_to_lower<float>(out, ncols);
    } else if (itemsize == 8) {
      launch_copy_upper_to_lower<double>(out, ncols);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });

  m.def("cov_from_gram", [](std::uintptr_t cov, std::uintptr_t gram, std::uintptr_t meanx,
                            std::uintptr_t meany, int ncols, int itemsize) {
    if (itemsize == 4) {
      launch_cov_from_gram<float>(cov, gram, meanx, meany, ncols);
    } else if (itemsize == 8) {
      launch_cov_from_gram<double>(cov, gram, meanx, meany, ncols);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });

  m.def("check_zero_genes", [](std::uintptr_t indices, std::uintptr_t genes, int nnz) {
    launch_check_zero_genes(indices, genes, nnz);
  });
}
