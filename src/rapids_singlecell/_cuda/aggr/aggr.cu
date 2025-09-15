#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

namespace nb = nanobind;

#include "kernels_aggr.cuh"

// Launchers
template <typename T>
static inline void launch_csr_aggr(std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
                                   std::uintptr_t out, std::uintptr_t cats, std::uintptr_t mask,
                                   std::size_t n_cells, std::size_t n_genes, std::size_t n_groups) {
  dim3 grid((unsigned)n_cells);
  dim3 block(64);
  csr_aggr_kernel<T>
      <<<grid, block>>>(reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
                        reinterpret_cast<const T*>(data), reinterpret_cast<double*>(out),
                        reinterpret_cast<const int*>(cats), reinterpret_cast<const bool*>(mask),
                        n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_csc_aggr(std::uintptr_t indptr, std::uintptr_t index, std::uintptr_t data,
                                   std::uintptr_t out, std::uintptr_t cats, std::uintptr_t mask,
                                   std::size_t n_cells, std::size_t n_genes, std::size_t n_groups) {
  dim3 grid((unsigned)n_genes);
  dim3 block(64);
  csc_aggr_kernel<T>
      <<<grid, block>>>(reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
                        reinterpret_cast<const T*>(data), reinterpret_cast<double*>(out),
                        reinterpret_cast<const int*>(cats), reinterpret_cast<const bool*>(mask),
                        n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_csr_to_coo(std::uintptr_t indptr, std::uintptr_t index,
                                     std::uintptr_t data, std::uintptr_t row, std::uintptr_t col,
                                     std::uintptr_t ndata, std::uintptr_t cats, std::uintptr_t mask,
                                     int n_cells) {
  dim3 grid((unsigned)n_cells);
  dim3 block(64);
  csr_to_coo_kernel<T><<<grid, block>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<const T*>(data), reinterpret_cast<int*>(row), reinterpret_cast<int*>(col),
      reinterpret_cast<double*>(ndata), reinterpret_cast<const int*>(cats),
      reinterpret_cast<const bool*>(mask), n_cells);
}

template <typename T>
static inline void launch_dense_C(std::uintptr_t data, std::uintptr_t out, std::uintptr_t cats,
                                  std::uintptr_t mask, std::size_t n_cells, std::size_t n_genes,
                                  std::size_t n_groups) {
  dim3 block(256);
  dim3 grid((unsigned)((n_cells * n_genes + block.x - 1) / block.x));
  dense_aggr_kernel_C<T>
      <<<grid, block>>>(reinterpret_cast<const T*>(data), reinterpret_cast<double*>(out),
                        reinterpret_cast<const int*>(cats), reinterpret_cast<const bool*>(mask),
                        n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_dense_F(std::uintptr_t data, std::uintptr_t out, std::uintptr_t cats,
                                  std::uintptr_t mask, std::size_t n_cells, std::size_t n_genes,
                                  std::size_t n_groups) {
  dim3 block(256);
  dim3 grid((unsigned)((n_cells * n_genes + block.x - 1) / block.x));
  dense_aggr_kernel_F<T>
      <<<grid, block>>>(reinterpret_cast<const T*>(data), reinterpret_cast<double*>(out),
                        reinterpret_cast<const int*>(cats), reinterpret_cast<const bool*>(mask),
                        n_cells, n_genes, n_groups);
}

// Unified dispatchers
static inline void sparse_aggr_dispatch(std::uintptr_t indptr, std::uintptr_t index,
                                        std::uintptr_t data, std::uintptr_t out,
                                        std::uintptr_t cats, std::uintptr_t mask,
                                        std::size_t n_cells, std::size_t n_genes,
                                        std::size_t n_groups, bool is_csc, int dtype_itemsize) {
  if (is_csc) {
    if (dtype_itemsize == 4) {
      launch_csc_aggr<float>(indptr, index, data, out, cats, mask, n_cells, n_genes, n_groups);
    } else {
      launch_csc_aggr<double>(indptr, index, data, out, cats, mask, n_cells, n_genes, n_groups);
    }
  } else {
    if (dtype_itemsize == 4) {
      launch_csr_aggr<float>(indptr, index, data, out, cats, mask, n_cells, n_genes, n_groups);
    } else {
      launch_csr_aggr<double>(indptr, index, data, out, cats, mask, n_cells, n_genes, n_groups);
    }
  }
}

static inline void dense_aggr_dispatch(std::uintptr_t data, std::uintptr_t out, std::uintptr_t cats,
                                       std::uintptr_t mask, std::size_t n_cells,
                                       std::size_t n_genes, std::size_t n_groups, bool is_fortran,
                                       int dtype_itemsize) {
  if (is_fortran) {
    if (dtype_itemsize == 4) {
      launch_dense_F<float>(data, out, cats, mask, n_cells, n_genes, n_groups);
    } else {
      launch_dense_F<double>(data, out, cats, mask, n_cells, n_genes, n_groups);
    }
  } else {
    if (dtype_itemsize == 4) {
      launch_dense_C<float>(data, out, cats, mask, n_cells, n_genes, n_groups);
    } else {
      launch_dense_C<double>(data, out, cats, mask, n_cells, n_genes, n_groups);
    }
  }
}

static inline void csr_to_coo_dispatch(std::uintptr_t indptr, std::uintptr_t index,
                                       std::uintptr_t data, std::uintptr_t row, std::uintptr_t col,
                                       std::uintptr_t ndata, std::uintptr_t cats,
                                       std::uintptr_t mask, int n_cells, int dtype_itemsize) {
  if (dtype_itemsize == 4) {
    launch_csr_to_coo<float>(indptr, index, data, row, col, ndata, cats, mask, n_cells);
  } else {
    launch_csr_to_coo<double>(indptr, index, data, row, col, ndata, cats, mask, n_cells);
  }
}

// variance launcher
static inline void launch_sparse_var(std::uintptr_t indptr, std::uintptr_t index,
                                     std::uintptr_t data, std::uintptr_t mean_data,
                                     std::uintptr_t n_cells, int dof, int n_groups) {
  dim3 grid((unsigned)n_groups);
  dim3 block(64);
  sparse_var_kernel<<<grid, block>>>(
      reinterpret_cast<const int*>(indptr), reinterpret_cast<const int*>(index),
      reinterpret_cast<double*>(data), reinterpret_cast<const double*>(mean_data),
      reinterpret_cast<double*>(n_cells), dof, n_groups);
}

NB_MODULE(_aggr_cuda, m) {
  m.def("sparse_aggr", &sparse_aggr_dispatch);
  m.def("dense_aggr", &dense_aggr_dispatch);
  m.def("csr_to_coo", &csr_to_coo_dispatch);
  m.def("sparse_var", &launch_sparse_var);
}
