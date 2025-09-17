#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_qcd.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_qc_csr_cells(std::uintptr_t indptr, std::uintptr_t index,
                                       std::uintptr_t data, std::uintptr_t sums_cells,
                                       std::uintptr_t cell_ex, int n_cells, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_cells + 31) / 32);
  qc_csr_cells_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<int*>(cell_ex), n_cells);
}

template <typename T>
static inline void launch_qc_csr_genes(std::uintptr_t index, std::uintptr_t data,
                                       std::uintptr_t sums_genes, std::uintptr_t gene_ex, int nnz,
                                       cudaStream_t stream) {
  int block = 256;
  int grid = (nnz + block - 1) / block;
  qc_csr_genes_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(index), reinterpret_cast<const T*>(data),
      reinterpret_cast<T*>(sums_genes), reinterpret_cast<int*>(gene_ex), nnz);
}

template <typename T>
static inline void launch_qc_dense_cells(std::uintptr_t data, std::uintptr_t sums_cells,
                                         std::uintptr_t cell_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n_cells + 15) / 16, (n_genes + 15) / 16);
  qc_dense_cells_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<int*>(cell_ex), n_cells, n_genes);
}

template <typename T>
static inline void launch_qc_dense_genes(std::uintptr_t data, std::uintptr_t sums_genes,
                                         std::uintptr_t gene_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n_cells + 15) / 16, (n_genes + 15) / 16);
  qc_dense_genes_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_genes),
      reinterpret_cast<int*>(gene_ex), n_cells, n_genes);
}

NB_MODULE(_qc_dask_cuda, m) {
  m.def(
      "sparse_qc_csr_cells",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t cell_ex, int n_cells, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_csr_cells<float>(indptr, index, data, sums_cells, cell_ex, n_cells,
                                     (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_csr_cells<double>(indptr, index, data, sums_cells, cell_ex, n_cells,
                                      (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("indptr"), nb::arg("index"), nb::arg("data"), nb::arg("sums_cells"),
      nb::arg("cell_ex"), nb::arg("n_cells"), nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_csr_genes",
      [](std::uintptr_t index, std::uintptr_t data, std::uintptr_t sums_genes,
         std::uintptr_t gene_ex, int nnz, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_csr_genes<float>(index, data, sums_genes, gene_ex, nnz, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_csr_genes<double>(index, data, sums_genes, gene_ex, nnz, (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("index"), nb::arg("data"), nb::arg("sums_genes"), nb::arg("gene_ex"), nb::arg("nnz"),
      nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_dense_cells",
      [](std::uintptr_t data, std::uintptr_t sums_cells, std::uintptr_t cell_ex, int n_cells,
         int n_genes, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_dense_cells<float>(data, sums_cells, cell_ex, n_cells, n_genes,
                                       (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_dense_cells<double>(data, sums_cells, cell_ex, n_cells, n_genes,
                                        (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("data"), nb::arg("sums_cells"), nb::arg("cell_ex"), nb::arg("n_cells"),
      nb::arg("n_genes"), nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_dense_genes",
      [](std::uintptr_t data, std::uintptr_t sums_genes, std::uintptr_t gene_ex, int n_cells,
         int n_genes, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_dense_genes<float>(data, sums_genes, gene_ex, n_cells, n_genes,
                                       (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_dense_genes<double>(data, sums_genes, gene_ex, n_cells, n_genes,
                                        (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("data"), nb::arg("sums_genes"), nb::arg("gene_ex"), nb::arg("n_cells"),
      nb::arg("n_genes"), nb::arg("itemsize"), nb::arg("stream") = 0);
}
