#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_aggr.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_csr_aggr(const int* indptr, const int* index,
                                   const T* data, double* out, const int* cats,
                                   const bool* mask, size_t n_cells,
                                   size_t n_genes, size_t n_groups,
                                   cudaStream_t stream) {
    dim3 grid((unsigned)n_cells);
    dim3 block(64);
    csr_aggr_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, out, cats, mask, n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_csc_aggr(const int* indptr, const int* index,
                                   const T* data, double* out, const int* cats,
                                   const bool* mask, size_t n_cells,
                                   size_t n_genes, size_t n_groups,
                                   cudaStream_t stream) {
    dim3 grid((unsigned)n_genes);
    dim3 block(64);
    csc_aggr_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, out, cats, mask, n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_dense_aggr_C(const T* data, double* out,
                                       const int* cats, const bool* mask,
                                       size_t n_cells, size_t n_genes,
                                       size_t n_groups, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((unsigned)((n_cells * n_genes + block.x - 1) / block.x));
    dense_aggr_kernel_C<T><<<grid, block, 0, stream>>>(
        data, out, cats, mask, n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_dense_aggr_F(const T* data, double* out,
                                       const int* cats, const bool* mask,
                                       size_t n_cells, size_t n_genes,
                                       size_t n_groups, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((unsigned)((n_cells * n_genes + block.x - 1) / block.x));
    dense_aggr_kernel_F<T><<<grid, block, 0, stream>>>(
        data, out, cats, mask, n_cells, n_genes, n_groups);
}

template <typename T>
static inline void launch_csr_to_coo(const int* indptr, const int* index,
                                     const T* data, int* row, int* col,
                                     double* ndata, const int* cats,
                                     const bool* mask, int n_cells,
                                     cudaStream_t stream) {
    dim3 grid((unsigned)n_cells);
    dim3 block(64);
    csr_to_coo_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, row, col, ndata, cats, mask, n_cells);
}

static inline void launch_sparse_var(const int* indptr, const int* index,
                                     double* data, const double* mean_data,
                                     double* n_cells, int dof, int n_groups,
                                     cudaStream_t stream) {
    dim3 grid((unsigned)n_groups);
    dim3 block(64);
    sparse_var_kernel<<<grid, block, 0, stream>>>(
        indptr, index, data, mean_data, n_cells, dof, n_groups);
}

template <typename T, typename Device>
void def_sparse_aggr(nb::module_& m) {
    m.def(
        "sparse_aggr",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> out,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const bool, Device> mask, size_t n_cells, size_t n_genes,
           size_t n_groups, bool is_csc, std::uintptr_t stream) {
            if (is_csc) {
                launch_csc_aggr<T>(indptr.data(), index.data(), data.data(),
                                   out.data(), cats.data(), mask.data(),
                                   n_cells, n_genes, n_groups,
                                   (cudaStream_t)stream);
            } else {
                launch_csr_aggr<T>(indptr.data(), index.data(), data.data(),
                                   out.data(), cats.data(), mask.data(),
                                   n_cells, n_genes, n_groups,
                                   (cudaStream_t)stream);
            }
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out"_a, "cats"_a,
        "mask"_a, "n_cells"_a, "n_genes"_a, "n_groups"_a, "is_csc"_a,
        "stream"_a = 0);
}

template <typename T, typename DataContig, typename Device>
void def_dense_aggr(nb::module_& m) {
    m.def(
        "dense_aggr",
        [](gpu_array_contig<const T, Device, DataContig> data,
           gpu_array_c<double, Device> out, gpu_array_c<const int, Device> cats,
           gpu_array_c<const bool, Device> mask, size_t n_cells, size_t n_genes,
           size_t n_groups, bool is_fortran, std::uintptr_t stream) {
            if constexpr (std::is_same_v<DataContig, nb::f_contig>) {
                launch_dense_aggr_F<T>(data.data(), out.data(), cats.data(),
                                       mask.data(), n_cells, n_genes, n_groups,
                                       (cudaStream_t)stream);
            } else {
                launch_dense_aggr_C<T>(data.data(), out.data(), cats.data(),
                                       mask.data(), n_cells, n_genes, n_groups,
                                       (cudaStream_t)stream);
            }
        },
        "data"_a, nb::kw_only(), "out"_a, "cats"_a, "mask"_a, "n_cells"_a,
        "n_genes"_a, "n_groups"_a, "is_fortran"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_csr_to_coo(nb::module_& m) {
    m.def(
        "csr_to_coo",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<int, Device> out_row,
           gpu_array_c<int, Device> out_col,
           gpu_array_c<double, Device> out_data,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const bool, Device> mask, int n_cells,
           std::uintptr_t stream) {
            launch_csr_to_coo<T>(indptr.data(), index.data(), data.data(),
                                 out_row.data(), out_col.data(),
                                 out_data.data(), cats.data(), mask.data(),
                                 n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out_row"_a,
        "out_col"_a, "out_data"_a, "cats"_a, "mask"_a, "n_cells"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_sparse_aggr<float, Device>(m);
    def_sparse_aggr<double, Device>(m);

    // F-order must come before C-order for proper dispatch
    def_dense_aggr<float, nb::f_contig, Device>(m);
    def_dense_aggr<float, nb::c_contig, Device>(m);
    def_dense_aggr<double, nb::f_contig, Device>(m);
    def_dense_aggr<double, nb::c_contig, Device>(m);

    def_csr_to_coo<float, Device>(m);
    def_csr_to_coo<double, Device>(m);

    m.def(
        "sparse_var",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<double, Device> data,
           gpu_array_c<const double, Device> means,
           gpu_array_c<double, Device> n_cells, int dof, int n_groups,
           std::uintptr_t stream) {
            launch_sparse_var(indptr.data(), index.data(), data.data(),
                              means.data(), n_cells.data(), dof, n_groups,
                              (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "n_cells"_a,
        "dof"_a, "n_groups"_a, "stream"_a = 0);
}

NB_MODULE(_aggr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
