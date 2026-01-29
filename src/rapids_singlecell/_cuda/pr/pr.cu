#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_pr.cuh"
#include "kernels_pr_hvg.cuh"

namespace nb = nanobind;
using namespace nb::literals;

// Templated cuda_array over dtype and contiguity
template <typename T, typename Contig = nb::c_contig>
using cuda_array = nb::ndarray<T, nb::device::cuda, Contig>;

template <typename T>
static inline void launch_sparse_norm_res_csc(const int* indptr, const int* index, const T* data,
                                              const T* sums_cells, const T* sums_genes,
                                              T* residuals, T inv_sum_total, T clip, T inv_theta,
                                              int n_cells, int n_genes, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  sparse_norm_res_csc_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, sums_cells,
                                                            sums_genes, residuals, inv_sum_total,
                                                            clip, inv_theta, n_cells, n_genes);
}

template <typename T>
static inline void launch_sparse_norm_res_csr(const int* indptr, const int* index, const T* data,
                                              const T* sums_cells, const T* sums_genes,
                                              T* residuals, T inv_sum_total, T clip, T inv_theta,
                                              int n_cells, int n_genes, cudaStream_t stream) {
  dim3 block(8);
  dim3 grid((n_cells + block.x - 1) / block.x);
  sparse_norm_res_csr_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, sums_cells,
                                                            sums_genes, residuals, inv_sum_total,
                                                            clip, inv_theta, n_cells, n_genes);
}

template <typename T>
static inline void launch_dense_norm_res(const T* X, T* residuals, const T* sums_cells,
                                         const T* sums_genes, T inv_sum_total, T clip, T inv_theta,
                                         int n_cells, int n_genes, cudaStream_t stream) {
  dim3 block(8, 8);
  dim3 grid((n_cells + block.x - 1) / block.x, (n_genes + block.y - 1) / block.y);
  dense_norm_res_kernel<T><<<grid, block, 0, stream>>>(
      X, residuals, sums_cells, sums_genes, inv_sum_total, clip, inv_theta, n_cells, n_genes);
}

template <typename T>
static inline void launch_sparse_sum_csc(const int* indptr, const int* index, const T* data,
                                         T* sums_genes, T* sums_cells, int n_genes,
                                         cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  sparse_sum_csc_kernel<T>
      <<<grid, block, 0, stream>>>(indptr, index, data, sums_genes, sums_cells, n_genes);
}

template <typename T>
static inline void launch_csc_hvg_res(const int* indptr, const int* index, const T* data,
                                      const T* sums_genes, const T* sums_cells, T* residuals,
                                      T inv_sum_total, T clip, T inv_theta, int n_genes,
                                      int n_cells, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  csc_hvg_res_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, sums_genes, sums_cells,
                                                    residuals, inv_sum_total, clip, inv_theta,
                                                    n_genes, n_cells);
}

template <typename T>
static inline void launch_dense_hvg_res(const T* data, const T* sums_genes, const T* sums_cells,
                                        T* residuals, T inv_sum_total, T clip, T inv_theta,
                                        int n_genes, int n_cells, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_genes + block.x - 1) / block.x);
  dense_hvg_res_kernel<T><<<grid, block, 0, stream>>>(
      data, sums_genes, sums_cells, residuals, inv_sum_total, clip, inv_theta, n_genes, n_cells);
}

// Helper to define sparse_norm_res_csc for a given dtype
template <typename T>
void def_sparse_norm_res_csc(nb::module_& m) {
  m.def(
      "sparse_norm_res_csc",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<const T> data,
         cuda_array<const T> sums_cells, cuda_array<const T> sums_genes, cuda_array<T> residuals,
         T inv_sum_total, T clip, T inv_theta, int n_cells, int n_genes, std::uintptr_t stream) {
        launch_sparse_norm_res_csc<T>(indptr.data(), index.data(), data.data(), sums_cells.data(),
                                      sums_genes.data(), residuals.data(), inv_sum_total, clip,
                                      inv_theta, n_cells, n_genes, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a, "sums_genes"_a, "residuals"_a,
      "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Helper to define sparse_norm_res_csr for a given dtype
template <typename T>
void def_sparse_norm_res_csr(nb::module_& m) {
  m.def(
      "sparse_norm_res_csr",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<const T> data,
         cuda_array<const T> sums_cells, cuda_array<const T> sums_genes, cuda_array<T> residuals,
         T inv_sum_total, T clip, T inv_theta, int n_cells, int n_genes, std::uintptr_t stream) {
        launch_sparse_norm_res_csr<T>(indptr.data(), index.data(), data.data(), sums_cells.data(),
                                      sums_genes.data(), residuals.data(), inv_sum_total, clip,
                                      inv_theta, n_cells, n_genes, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a, "sums_genes"_a, "residuals"_a,
      "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Helper to define dense_norm_res for a given dtype
template <typename T>
void def_dense_norm_res(nb::module_& m) {
  m.def(
      "dense_norm_res",
      [](cuda_array<const T> X, cuda_array<T> residuals, cuda_array<const T> sums_cells,
         cuda_array<const T> sums_genes, T inv_sum_total, T clip, T inv_theta, int n_cells,
         int n_genes, std::uintptr_t stream) {
        launch_dense_norm_res<T>(X.data(), residuals.data(), sums_cells.data(), sums_genes.data(),
                                 inv_sum_total, clip, inv_theta, n_cells, n_genes,
                                 (cudaStream_t)stream);
      },
      "X"_a, nb::kw_only(), "residuals"_a, "sums_cells"_a, "sums_genes"_a, "inv_sum_total"_a,
      "clip"_a, "inv_theta"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Helper to define sparse_sum_csc for a given dtype
template <typename T>
void def_sparse_sum_csc(nb::module_& m) {
  m.def(
      "sparse_sum_csc",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<const T> data,
         cuda_array<T> sums_genes, cuda_array<T> sums_cells, int n_genes, std::uintptr_t stream) {
        launch_sparse_sum_csc<T>(indptr.data(), index.data(), data.data(), sums_genes.data(),
                                 sums_cells.data(), n_genes, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "sums_cells"_a, "n_genes"_a,
      "stream"_a = 0);
}

// Helper to define csc_hvg_res for a given dtype
template <typename T>
void def_csc_hvg_res(nb::module_& m) {
  m.def(
      "csc_hvg_res",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<const T> data,
         cuda_array<const T> sums_genes, cuda_array<const T> sums_cells, cuda_array<T> residuals,
         T inv_sum_total, T clip, T inv_theta, int n_genes, int n_cells, std::uintptr_t stream) {
        launch_csc_hvg_res<T>(indptr.data(), index.data(), data.data(), sums_genes.data(),
                              sums_cells.data(), residuals.data(), inv_sum_total, clip, inv_theta,
                              n_genes, n_cells, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "sums_cells"_a, "residuals"_a,
      "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_genes"_a, "n_cells"_a, "stream"_a = 0);
}

// Helper to define dense_hvg_res for a given dtype (always F-contiguous input)
template <typename T>
void def_dense_hvg_res(nb::module_& m) {
  m.def(
      "dense_hvg_res",
      [](cuda_array<const T, nb::f_contig> data, cuda_array<const T> sums_genes,
         cuda_array<const T> sums_cells, cuda_array<T> residuals, T inv_sum_total, T clip,
         T inv_theta, int n_genes, int n_cells, std::uintptr_t stream) {
        launch_dense_hvg_res<T>(data.data(), sums_genes.data(), sums_cells.data(), residuals.data(),
                                inv_sum_total, clip, inv_theta, n_genes, n_cells,
                                (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "sums_genes"_a, "sums_cells"_a, "residuals"_a, "inv_sum_total"_a,
      "clip"_a, "inv_theta"_a, "n_genes"_a, "n_cells"_a, "stream"_a = 0);
}

NB_MODULE(_pr_cuda, m) {
  // sparse_norm_res_csc
  def_sparse_norm_res_csc<float>(m);
  def_sparse_norm_res_csc<double>(m);

  // sparse_norm_res_csr
  def_sparse_norm_res_csr<float>(m);
  def_sparse_norm_res_csr<double>(m);

  // dense_norm_res
  def_dense_norm_res<float>(m);
  def_dense_norm_res<double>(m);

  // sparse_sum_csc
  def_sparse_sum_csc<float>(m);
  def_sparse_sum_csc<double>(m);

  // csc_hvg_res
  def_csc_hvg_res<float>(m);
  def_csc_hvg_res<double>(m);

  // dense_hvg_res - always F-contiguous (Python calls cp.asfortranarray)
  def_dense_hvg_res<float>(m);
  def_dense_hvg_res<double>(m);
}
