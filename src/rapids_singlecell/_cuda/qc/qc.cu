#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_qc.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_qc_csc(std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
                                 std::uintptr_t sums_cells, std::uintptr_t sums_genes,
                                 std::uintptr_t cell_ex, std::uintptr_t gene_ex, int n_genes,
                                 cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  qc_csc_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<T*>(sums_genes), reinterpret_cast<int*>(cell_ex),
      reinterpret_cast<int*>(gene_ex), n_genes);
}

template <typename T>
static inline void launch_qc_csr(std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
                                 std::uintptr_t sums_cells, std::uintptr_t sums_genes,
                                 std::uintptr_t cell_ex, std::uintptr_t gene_ex, int n_cells,
                                 cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_cells + block.x - 1) / block.x);
  qc_csr_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<T*>(sums_genes), reinterpret_cast<int*>(cell_ex),
      reinterpret_cast<int*>(gene_ex), n_cells);
}

template <typename T>
static inline void launch_qc_dense(std::uintptr_t data, std::uintptr_t sums_cells,
                                   std::uintptr_t sums_genes, std::uintptr_t cell_ex,
                                   std::uintptr_t gene_ex, int n_cells, int n_genes,
                                   cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n_cells + block.x - 1) / block.x, (n_genes + block.y - 1) / block.y);
  qc_dense_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<T*>(sums_genes), reinterpret_cast<int*>(cell_ex),
      reinterpret_cast<int*>(gene_ex), n_cells, n_genes);
}

template <typename T>
static inline void launch_qc_csc_sub(std::uintptr_t indptr, std::uintptr_t index,
                                     std::uintptr_t data, std::uintptr_t sums_cells,
                                     std::uintptr_t mask, int n_genes, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  qc_csc_sub_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<const bool*>(mask), n_genes);
}

template <typename T>
static inline void launch_qc_csr_sub(std::uintptr_t indptr, std::uintptr_t index,
                                     std::uintptr_t data, std::uintptr_t sums_cells,
                                     std::uintptr_t mask, int n_cells, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_cells + block.x - 1) / block.x);
  qc_csr_sub_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<const bool*>(mask), n_cells);
}

template <typename T>
static inline void launch_qc_dense_sub(std::uintptr_t data, std::uintptr_t sums_cells,
                                       std::uintptr_t mask, int n_cells, int n_genes,
                                       cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n_cells + block.x - 1) / block.x, (n_genes + block.y - 1) / block.y);
  qc_dense_sub_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(data), reinterpret_cast<T*>(sums_cells),
      reinterpret_cast<const bool*>(mask), n_cells, n_genes);
}

NB_MODULE(_qc_cuda, m) {
  m.def(
      "sparse_qc_csc",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t sums_genes, std::uintptr_t cell_ex,
         std::uintptr_t gene_ex, int n_genes, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_csc<float>(indptr, index, data, sums_cells, sums_genes, cell_ex, gene_ex,
                               n_genes, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_csc<double>(indptr, index, data, sums_cells, sums_genes, cell_ex, gene_ex,
                                n_genes, (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("indptr"), nb::arg("index"), nb::arg("data"), nb::arg("sums_cells"),
      nb::arg("sums_genes"), nb::arg("cell_ex"), nb::arg("gene_ex"), nb::arg("n_genes"),
      nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_csr",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t sums_genes, std::uintptr_t cell_ex,
         std::uintptr_t gene_ex, int n_cells, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_csr<float>(indptr, index, data, sums_cells, sums_genes, cell_ex, gene_ex,
                               n_cells, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_csr<double>(indptr, index, data, sums_cells, sums_genes, cell_ex, gene_ex,
                                n_cells, (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("indptr"), nb::arg("index"), nb::arg("data"), nb::arg("sums_cells"),
      nb::arg("sums_genes"), nb::arg("cell_ex"), nb::arg("gene_ex"), nb::arg("n_cells"),
      nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_dense",
      [](std::uintptr_t data, std::uintptr_t sums_cells, std::uintptr_t sums_genes,
         std::uintptr_t cell_ex, std::uintptr_t gene_ex, int n_cells, int n_genes, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_dense<float>(data, sums_cells, sums_genes, cell_ex, gene_ex, n_cells, n_genes,
                                 (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_dense<double>(data, sums_cells, sums_genes, cell_ex, gene_ex, n_cells, n_genes,
                                  (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("data"), nb::arg("sums_cells"), nb::arg("sums_genes"), nb::arg("cell_ex"),
      nb::arg("gene_ex"), nb::arg("n_cells"), nb::arg("n_genes"), nb::arg("itemsize"),
      nb::arg("stream") = 0);
  m.def(
      "sparse_qc_csc_sub",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t mask, int n_genes, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_csc_sub<float>(indptr, index, data, sums_cells, mask, n_genes,
                                   (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_csc_sub<double>(indptr, index, data, sums_cells, mask, n_genes,
                                    (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("indptr"), nb::arg("index"), nb::arg("data"), nb::arg("sums_cells"), nb::arg("mask"),
      nb::arg("n_genes"), nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_csr_sub",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t mask, int n_cells, int itemsize,
         std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_csr_sub<float>(indptr, index, data, sums_cells, mask, n_cells,
                                   (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_csr_sub<double>(indptr, index, data, sums_cells, mask, n_cells,
                                    (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("indptr"), nb::arg("index"), nb::arg("data"), nb::arg("sums_cells"), nb::arg("mask"),
      nb::arg("n_cells"), nb::arg("itemsize"), nb::arg("stream") = 0);
  m.def(
      "sparse_qc_dense_sub",
      [](std::uintptr_t data, std::uintptr_t sums_cells, std::uintptr_t mask, int n_cells,
         int n_genes, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_qc_dense_sub<float>(data, sums_cells, mask, n_cells, n_genes,
                                     (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_qc_dense_sub<double>(data, sums_cells, mask, n_cells, n_genes,
                                      (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      nb::arg("data"), nb::arg("sums_cells"), nb::arg("mask"), nb::arg("n_cells"),
      nb::arg("n_genes"), nb::arg("itemsize"), nb::arg("stream") = 0);
}
