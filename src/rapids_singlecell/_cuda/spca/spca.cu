#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>
#include <string>

#include "kernels_spca.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
static inline void launch_gram_csr_upper(std::uintptr_t indptr_ptr, std::uintptr_t index_ptr,
                                         std::uintptr_t data_ptr, int nrows, int ncols,
                                         std::uintptr_t out_ptr, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid(nrows);
  const int* indptr = reinterpret_cast<const int*>(indptr_ptr);
  const int* index = reinterpret_cast<const int*>(index_ptr);
  const T* data = reinterpret_cast<const T*>(data_ptr);
  T* out = reinterpret_cast<T*>(out_ptr);
  gram_csr_upper_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, nrows, ncols, out);
}

template <typename T>
static inline void launch_copy_upper_to_lower(std::uintptr_t out_ptr, int ncols,
                                              cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((ncols + block.x - 1) / block.x, (ncols + block.y - 1) / block.y);
  T* out = reinterpret_cast<T*>(out_ptr);
  copy_upper_to_lower_kernel<T><<<grid, block, 0, stream>>>(out, ncols);
}

template <typename T>
static inline void launch_cov_from_gram(std::uintptr_t cov_ptr, std::uintptr_t gram_ptr,
                                        std::uintptr_t meanx_ptr, std::uintptr_t meany_ptr,
                                        int ncols, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((ncols + 31) / 32, (ncols + 31) / 32);
  T* cov = reinterpret_cast<T*>(cov_ptr);
  const T* gram = reinterpret_cast<const T*>(gram_ptr);
  const T* meanx = reinterpret_cast<const T*>(meanx_ptr);
  const T* meany = reinterpret_cast<const T*>(meany_ptr);
  cov_from_gram_kernel<T><<<grid, block, 0, stream>>>(cov, gram, meanx, meany, ncols);
}

static inline void launch_check_zero_genes(std::uintptr_t indices_ptr, std::uintptr_t genes_ptr,
                                           int nnz, int num_genes, cudaStream_t stream) {
  if (nnz > 0) {
    dim3 block(32);
    dim3 grid((nnz + block.x - 1) / block.x);
    const int* indices = reinterpret_cast<const int*>(indices_ptr);
    int* genes = reinterpret_cast<int*>(genes_ptr);
    check_zero_genes_kernel<<<grid, block, 0, stream>>>(indices, genes, nnz, num_genes);
  }
}

NB_MODULE(_spca_cuda, m) {
  m.def(
      "gram_csr_upper",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data, int nrows, int ncols,
         std::uintptr_t out, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_gram_csr_upper<float>(indptr, index, data, nrows, ncols, out,
                                       (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_gram_csr_upper<double>(indptr, index, data, nrows, ncols, out,
                                        (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "indptr"_a, "index"_a, "data"_a, "nrows"_a, "ncols"_a, "out"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "copy_upper_to_lower",
      [](std::uintptr_t out, int ncols, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_copy_upper_to_lower<float>(out, ncols, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_copy_upper_to_lower<double>(out, ncols, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "out"_a, "ncols"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "cov_from_gram",
      [](std::uintptr_t cov, std::uintptr_t gram, std::uintptr_t meanx, std::uintptr_t meany,
         int ncols, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_cov_from_gram<float>(cov, gram, meanx, meany, ncols, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_cov_from_gram<double>(cov, gram, meanx, meany, ncols, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      "cov"_a, "gram"_a, "meanx"_a, "meany"_a, "ncols"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "check_zero_genes",
      [](std::uintptr_t indices, std::uintptr_t genes, int nnz, int num_genes,
         std::uintptr_t stream) {
        launch_check_zero_genes(indices, genes, nnz, num_genes, (cudaStream_t)stream);
      },
      "indices"_a, "genes"_a, "nnz"_a, "num_genes"_a, "stream"_a = 0);
}
