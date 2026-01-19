#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_aggr.cuh"

namespace nb = nanobind;
using namespace nb::literals;

// Templated cuda_array over dtype and contiguity
template <typename T, typename Contig = nb::c_contig>
using cuda_array = nb::ndarray<T, nb::device::cuda, Contig>;

template <typename T>
static inline void launch_csr_aggr(const int* indptr, const int* index, const T* data, double* out,
                                   const int* cats, const bool* mask, std::size_t n_cells,
                                   std::size_t n_genes, std::size_t n_groups, cudaStream_t stream) {
  dim3 grid((unsigned)n_cells);
  dim3 block(64);
  csr_aggr_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, out, cats, mask, n_cells,
                                                 n_genes, n_groups);
}

template <typename T>
static inline void launch_csc_aggr(const int* indptr, const int* index, const T* data, double* out,
                                   const int* cats, const bool* mask, std::size_t n_cells,
                                   std::size_t n_genes, std::size_t n_groups, cudaStream_t stream) {
  dim3 grid((unsigned)n_genes);
  dim3 block(64);
  csc_aggr_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data, out, cats, mask, n_cells,
                                                 n_genes, n_groups);
}

template <typename T>
static inline void launch_csr_to_coo(const int* indptr, const int* index, const T* data, int* row,
                                     int* col, double* ndata, const int* cats, const bool* mask,
                                     int n_cells, cudaStream_t stream) {
  dim3 grid((unsigned)n_cells);
  dim3 block(64);
  csr_to_coo_kernel<T>
      <<<grid, block, 0, stream>>>(indptr, index, data, row, col, ndata, cats, mask, n_cells);
}

template <typename T>
static inline void launch_dense_C(const T* data, double* out, const int* cats, const bool* mask,
                                  std::size_t n_cells, std::size_t n_genes, std::size_t n_groups,
                                  cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((unsigned)((n_cells * n_genes + block.x - 1) / block.x));
  dense_aggr_kernel_C<T>
      <<<grid, block, 0, stream>>>(data, out, cats, mask, n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_dense_F(const T* data, double* out, const int* cats, const bool* mask,
                                  std::size_t n_cells, std::size_t n_genes, std::size_t n_groups,
                                  cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((unsigned)((n_cells * n_genes + block.x - 1) / block.x));
  dense_aggr_kernel_F<T>
      <<<grid, block, 0, stream>>>(data, out, cats, mask, n_cells, n_genes, n_groups);
}

static inline void launch_sparse_var(const int* indptr, const int* index, double* data,
                                     const double* mean_data, double* n_cells, int dof,
                                     int n_groups, cudaStream_t stream) {
  dim3 grid((unsigned)n_groups);
  dim3 block(64);
  sparse_var_kernel<<<grid, block, 0, stream>>>(indptr, index, data, mean_data, n_cells, dof,
                                                n_groups);
}

// Helper to define sparse_aggr for a given dtype
template <typename T>
void def_sparse_aggr(nb::module_& m) {
  m.def(
      "sparse_aggr",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<const T> data,
         cuda_array<double> out, cuda_array<const int> cats, cuda_array<const bool> mask,
         std::size_t n_cells, std::size_t n_genes, std::size_t n_groups, bool is_csc,
         std::uintptr_t stream) {
        if (is_csc) {
          launch_csc_aggr<T>(indptr.data(), index.data(), data.data(), out.data(), cats.data(),
                             mask.data(), n_cells, n_genes, n_groups, (cudaStream_t)stream);
        } else {
          launch_csr_aggr<T>(indptr.data(), index.data(), data.data(), out.data(), cats.data(),
                             mask.data(), n_cells, n_genes, n_groups, (cudaStream_t)stream);
        }
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out"_a, "cats"_a, "mask"_a, "n_cells"_a,
      "n_genes"_a, "n_groups"_a, "is_csc"_a, "stream"_a = 0);
}

// Helper to define dense_aggr for a given dtype and contiguity
template <typename T, typename DataContig, bool IsFortran>
void def_dense_aggr(nb::module_& m) {
  m.def(
      "dense_aggr",
      [](cuda_array<const T, DataContig> data, cuda_array<double> out, cuda_array<const int> cats,
         cuda_array<const bool> mask, std::size_t n_cells, std::size_t n_genes,
         std::size_t n_groups, bool is_fortran, std::uintptr_t stream) {
        if constexpr (IsFortran) {
          launch_dense_F<T>(data.data(), out.data(), cats.data(), mask.data(), n_cells, n_genes,
                            n_groups, (cudaStream_t)stream);
        } else {
          launch_dense_C<T>(data.data(), out.data(), cats.data(), mask.data(), n_cells, n_genes,
                            n_groups, (cudaStream_t)stream);
        }
      },
      "data"_a, nb::kw_only(), "out"_a, "cats"_a, "mask"_a, "n_cells"_a, "n_genes"_a, "n_groups"_a,
      "is_fortran"_a, "stream"_a = 0);
}

// Helper to define csr_to_coo for a given dtype
template <typename T>
void def_csr_to_coo(nb::module_& m) {
  m.def(
      "csr_to_coo",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<const T> data,
         cuda_array<int> out_row, cuda_array<int> out_col, cuda_array<double> out_data,
         cuda_array<const int> cats, cuda_array<const bool> mask, int n_cells,
         std::uintptr_t stream) {
        launch_csr_to_coo<T>(indptr.data(), index.data(), data.data(), out_row.data(),
                             out_col.data(), out_data.data(), cats.data(), mask.data(), n_cells,
                             (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out_row"_a, "out_col"_a, "out_data"_a,
      "cats"_a, "mask"_a, "n_cells"_a, "stream"_a = 0);
}

NB_MODULE(_aggr_cuda, m) {
  // sparse_aggr
  def_sparse_aggr<float>(m);
  def_sparse_aggr<double>(m);

  // dense_aggr - F-order must come before C-order for proper dispatch
  def_dense_aggr<float, nb::f_contig, true>(m);
  def_dense_aggr<float, nb::c_contig, false>(m);
  def_dense_aggr<double, nb::f_contig, true>(m);
  def_dense_aggr<double, nb::c_contig, false>(m);

  // csr_to_coo
  def_csr_to_coo<float>(m);
  def_csr_to_coo<double>(m);

  // sparse_var
  m.def(
      "sparse_var",
      [](cuda_array<const int> indptr, cuda_array<const int> index, cuda_array<double> data,
         cuda_array<const double> means, cuda_array<double> n_cells, int dof, int n_groups,
         std::uintptr_t stream) {
        launch_sparse_var(indptr.data(), index.data(), data.data(), means.data(), n_cells.data(),
                          dof, n_groups, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "n_cells"_a, "dof"_a, "n_groups"_a,
      "stream"_a = 0);
}
