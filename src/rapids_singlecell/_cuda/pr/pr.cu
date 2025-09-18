#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_pr.cuh"
#include "kernels_pr_hvg.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
static inline void launch_sparse_norm_res_csc(std::uintptr_t indptr, std::uintptr_t index,
                                              std::uintptr_t data, std::uintptr_t sums_cells,
                                              std::uintptr_t sums_genes, std::uintptr_t residuals,
                                              T inv_sum_total, T clip, T inv_theta, int n_cells,
                                              int n_genes, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  sparse_norm_res_csc_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<const T*>(sums_cells),
      reinterpret_cast<const T*>(sums_genes), reinterpret_cast<T*>(residuals), inv_sum_total, clip,
      inv_theta, n_cells, n_genes);
}

template <typename T>
static inline void launch_sparse_norm_res_csr(std::uintptr_t indptr, std::uintptr_t index,
                                              std::uintptr_t data, std::uintptr_t sums_cells,
                                              std::uintptr_t sums_genes, std::uintptr_t residuals,
                                              T inv_sum_total, T clip, T inv_theta, int n_cells,
                                              int n_genes, cudaStream_t stream) {
  dim3 block(8);
  dim3 grid((n_cells + block.x - 1) / block.x);
  sparse_norm_res_csr_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<const T*>(sums_cells),
      reinterpret_cast<const T*>(sums_genes), reinterpret_cast<T*>(residuals), inv_sum_total, clip,
      inv_theta, n_cells, n_genes);
}

template <typename T>
static inline void launch_dense_norm_res(std::uintptr_t X, std::uintptr_t residuals,
                                         std::uintptr_t sums_cells, std::uintptr_t sums_genes,
                                         T inv_sum_total, T clip, T inv_theta, int n_cells,
                                         int n_genes, cudaStream_t stream) {
  dim3 block(8, 8);
  dim3 grid((n_cells + block.x - 1) / block.x, (n_genes + block.y - 1) / block.y);
  dense_norm_res_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(X), reinterpret_cast<T*>(residuals),
      reinterpret_cast<const T*>(sums_cells), reinterpret_cast<const T*>(sums_genes), inv_sum_total,
      clip, inv_theta, n_cells, n_genes);
}

template <typename T>
static inline void launch_csc_hvg_res(std::uintptr_t indptr, std::uintptr_t index,
                                      std::uintptr_t data, std::uintptr_t sums_genes,
                                      std::uintptr_t sums_cells, std::uintptr_t residuals,
                                      T inv_sum_total, T clip, T inv_theta, int n_genes,
                                      int n_cells, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  csc_hvg_res_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<const T*>(sums_genes),
      reinterpret_cast<const T*>(sums_cells), reinterpret_cast<T*>(residuals), inv_sum_total, clip,
      inv_theta, n_genes, n_cells);
}

template <typename T>
static inline void launch_dense_hvg_res(std::uintptr_t data, std::uintptr_t sums_genes,
                                        std::uintptr_t sums_cells, std::uintptr_t residuals,
                                        T inv_sum_total, T clip, T inv_theta, int n_genes,
                                        int n_cells, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  dense_hvg_res_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(data), reinterpret_cast<const T*>(sums_genes),
      reinterpret_cast<const T*>(sums_cells), reinterpret_cast<T*>(residuals), inv_sum_total, clip,
      inv_theta, n_genes, n_cells);
}

NB_MODULE(_pr_cuda, m) {
  m.def(
      "sparse_norm_res_csc",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t sums_genes, std::uintptr_t residuals,
         double inv_sum_total, double clip, double inv_theta, int n_cells, int n_genes,
         int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_sparse_norm_res_csc<float>(indptr, index, data, sums_cells, sums_genes, residuals,
                                            (float)inv_sum_total, (float)clip, (float)inv_theta,
                                            n_cells, n_genes, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_sparse_norm_res_csc<double>(indptr, index, data, sums_cells, sums_genes, residuals,
                                             inv_sum_total, clip, inv_theta, n_cells, n_genes,
                                             (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      "indptr"_a, "index"_a, "data"_a, "sums_cells"_a, "sums_genes"_a, "residuals"_a,
      "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_cells"_a, "n_genes"_a, "itemsize"_a,
      "stream"_a = 0);

  m.def(
      "sparse_norm_res_csr",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_cells, std::uintptr_t sums_genes, std::uintptr_t residuals,
         double inv_sum_total, double clip, double inv_theta, int n_cells, int n_genes,
         int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_sparse_norm_res_csr<float>(indptr, index, data, sums_cells, sums_genes, residuals,
                                            (float)inv_sum_total, (float)clip, (float)inv_theta,
                                            n_cells, n_genes, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_sparse_norm_res_csr<double>(indptr, index, data, sums_cells, sums_genes, residuals,
                                             inv_sum_total, clip, inv_theta, n_cells, n_genes,
                                             (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      "indptr"_a, "index"_a, "data"_a, "sums_cells"_a, "sums_genes"_a, "residuals"_a,
      "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_cells"_a, "n_genes"_a, "itemsize"_a,
      "stream"_a = 0);

  m.def(
      "dense_norm_res",
      [](std::uintptr_t X, std::uintptr_t residuals, std::uintptr_t sums_cells,
         std::uintptr_t sums_genes, double inv_sum_total, double clip, double inv_theta,
         int n_cells, int n_genes, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_dense_norm_res<float>(X, residuals, sums_cells, sums_genes, (float)inv_sum_total,
                                       (float)clip, (float)inv_theta, n_cells, n_genes,
                                       (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_dense_norm_res<double>(X, residuals, sums_cells, sums_genes, inv_sum_total, clip,
                                        inv_theta, n_cells, n_genes, (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      "X"_a, "residuals"_a, "sums_cells"_a, "sums_genes"_a, "inv_sum_total"_a, "clip"_a,
      "inv_theta"_a, "n_cells"_a, "n_genes"_a, "itemsize"_a, "stream"_a = 0);

  m.def(
      "csc_hvg_res",
      [](std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
         std::uintptr_t sums_genes, std::uintptr_t sums_cells, std::uintptr_t residuals,
         double inv_sum_total, double clip, double inv_theta, int n_genes, int n_cells,
         int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_csc_hvg_res<float>(indptr, index, data, sums_genes, sums_cells, residuals,
                                    (float)inv_sum_total, (float)clip, (float)inv_theta, n_genes,
                                    n_cells, (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_csc_hvg_res<double>(indptr, index, data, sums_genes, sums_cells, residuals,
                                     inv_sum_total, clip, inv_theta, n_genes, n_cells,
                                     (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      "indptr"_a, "index"_a, "data"_a, "sums_genes"_a, "sums_cells"_a, "residuals"_a,
      "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_genes"_a, "n_cells"_a, "itemsize"_a,
      "stream"_a = 0);

  m.def(
      "dense_hvg_res",
      [](std::uintptr_t data, std::uintptr_t sums_genes, std::uintptr_t sums_cells,
         std::uintptr_t residuals, double inv_sum_total, double clip, double inv_theta, int n_genes,
         int n_cells, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4)
          launch_dense_hvg_res<float>(data, sums_genes, sums_cells, residuals, (float)inv_sum_total,
                                      (float)clip, (float)inv_theta, n_genes, n_cells,
                                      (cudaStream_t)stream);
        else if (itemsize == 8)
          launch_dense_hvg_res<double>(data, sums_genes, sums_cells, residuals, inv_sum_total, clip,
                                       inv_theta, n_genes, n_cells, (cudaStream_t)stream);
        else
          throw nb::value_error("Unsupported itemsize");
      },
      "data"_a, "sums_genes"_a, "sums_cells"_a, "residuals"_a, "inv_sum_total"_a, "clip"_a,
      "inv_theta"_a, "n_genes"_a, "n_cells"_a, "itemsize"_a, "stream"_a = 0);
}
