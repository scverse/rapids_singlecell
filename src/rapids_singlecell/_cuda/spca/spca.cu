#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_spca.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_gram_csr_upper(const int* indptr, const int* index, const T* data,
                                         int nrows, int ncols, T* out, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid(nrows);
  gram_csr_upper_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, nrows, ncols, out);
}

template <typename T>
static inline void launch_copy_upper_to_lower(T* out, int ncols, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((ncols + block.x - 1) / block.x, (ncols + block.y - 1) / block.y);
  copy_upper_to_lower_kernel<T><<<grid, block, 0, stream>>>(out, ncols);
}

template <typename T>
static inline void launch_cov_from_gram(T* cov, const T* gram, const T* meanx, const T* meany,
                                        int ncols, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((ncols + 31) / 32, (ncols + 31) / 32);
  cov_from_gram_kernel<T><<<grid, block, 0, stream>>>(cov, gram, meanx, meany, ncols);
}

static inline void launch_check_zero_genes(const int* indices, int* genes, int nnz, int num_genes,
                                           cudaStream_t stream) {
  if (nnz > 0) {
    dim3 block(32);
    dim3 grid((nnz + block.x - 1) / block.x);
    check_zero_genes_kernel<<<grid, block, 0, stream>>>(indices, genes, nnz, num_genes);
  }
}

NB_MODULE(_spca_cuda, m) {
  // gram_csr_upper - float32
  m.def(
      "gram_csr_upper",
      [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
         cuda_array_c<const float> data, int nrows, int ncols, cuda_array_c<float> out,
         std::uintptr_t stream) {
        launch_gram_csr_upper<float>(indptr.data(), index.data(), data.data(), nrows, ncols,
                                     out.data(), (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "out"_a,
      "stream"_a = 0);

  // gram_csr_upper - float64
  m.def(
      "gram_csr_upper",
      [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
         cuda_array_c<const double> data, int nrows, int ncols, cuda_array_c<double> out,
         std::uintptr_t stream) {
        launch_gram_csr_upper<double>(indptr.data(), index.data(), data.data(), nrows, ncols,
                                      out.data(), (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "out"_a,
      "stream"_a = 0);

  // copy_upper_to_lower - float32
  m.def(
      "copy_upper_to_lower",
      [](cuda_array_c<float> out, int ncols, std::uintptr_t stream) {
        launch_copy_upper_to_lower<float>(out.data(), ncols, (cudaStream_t)stream);
      },
      nb::kw_only(), "out"_a, "ncols"_a, "stream"_a = 0);

  // copy_upper_to_lower - float64
  m.def(
      "copy_upper_to_lower",
      [](cuda_array_c<double> out, int ncols, std::uintptr_t stream) {
        launch_copy_upper_to_lower<double>(out.data(), ncols, (cudaStream_t)stream);
      },
      nb::kw_only(), "out"_a, "ncols"_a, "stream"_a = 0);

  // cov_from_gram - float32
  m.def(
      "cov_from_gram",
      [](cuda_array_c<const float> gram, cuda_array_c<const float> meanx,
         cuda_array_c<const float> meany, cuda_array_c<float> cov, int ncols,
         std::uintptr_t stream) {
        launch_cov_from_gram<float>(cov.data(), gram.data(), meanx.data(), meany.data(), ncols,
                                    (cudaStream_t)stream);
      },
      "gram"_a, "meanx"_a, "meany"_a, nb::kw_only(), "cov"_a, "ncols"_a, "stream"_a = 0);

  // cov_from_gram - float64
  m.def(
      "cov_from_gram",
      [](cuda_array_c<const double> gram, cuda_array_c<const double> meanx,
         cuda_array_c<const double> meany, cuda_array_c<double> cov, int ncols,
         std::uintptr_t stream) {
        launch_cov_from_gram<double>(cov.data(), gram.data(), meanx.data(), meany.data(), ncols,
                                     (cudaStream_t)stream);
      },
      "gram"_a, "meanx"_a, "meany"_a, nb::kw_only(), "cov"_a, "ncols"_a, "stream"_a = 0);

  // check_zero_genes
  m.def(
      "check_zero_genes",
      [](cuda_array_c<const int> indices, cuda_array_c<int> out, int nnz, int num_genes,
         std::uintptr_t stream) {
        launch_check_zero_genes(indices.data(), out.data(), nnz, num_genes, (cudaStream_t)stream);
      },
      "indices"_a, nb::kw_only(), "out"_a, "nnz"_a, "num_genes"_a, "stream"_a = 0);
}
