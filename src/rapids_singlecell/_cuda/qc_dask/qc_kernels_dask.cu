#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_qcd.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_qc_csr_cells(const int* indptr, const int* index,
                                       const T* data, T* sums_cells,
                                       int* cell_ex, int n_cells,
                                       cudaStream_t stream) {
    dim3 block(32);
    dim3 grid((n_cells + 31) / 32);
    qc_csr_cells_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, cell_ex, n_cells);
}

template <typename T>
static inline void launch_qc_csr_genes(const int* index, const T* data,
                                       T* sums_genes, int* gene_ex, int nnz,
                                       cudaStream_t stream) {
    int block = 256;
    int grid = (nnz + block - 1) / block;
    qc_csr_genes_kernel<T>
        <<<grid, block, 0, stream>>>(index, data, sums_genes, gene_ex, nnz);
}

template <typename T>
static inline void launch_qc_dense_cells(const T* data, T* sums_cells,
                                         int* cell_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n_cells + 15) / 16, (n_genes + 15) / 16);
    qc_dense_cells_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_cells, cell_ex, n_cells, n_genes);
}

template <typename T>
static inline void launch_qc_dense_genes(const T* data, T* sums_genes,
                                         int* gene_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n_cells + 15) / 16, (n_genes + 15) / 16);
    qc_dense_genes_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_genes, gene_ex, n_cells, n_genes);
}

NB_MODULE(_qc_dask_cuda, m) {
    // sparse_qc_csr_cells - float32
    m.def(
        "sparse_qc_csr_cells",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
           cuda_array_c<const float> data, cuda_array_c<float> sums_cells,
           cuda_array_c<int> cell_ex, int n_cells, std::uintptr_t stream) {
            launch_qc_csr_cells<float>(indptr.data(), index.data(), data.data(),
                                       sums_cells.data(), cell_ex.data(),
                                       n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "cell_ex"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_csr_cells - float64
    m.def(
        "sparse_qc_csr_cells",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> index,
           cuda_array_c<const double> data, cuda_array_c<double> sums_cells,
           cuda_array_c<int> cell_ex, int n_cells, std::uintptr_t stream) {
            launch_qc_csr_cells<double>(
                indptr.data(), index.data(), data.data(), sums_cells.data(),
                cell_ex.data(), n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "cell_ex"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_csr_genes - float32
    m.def(
        "sparse_qc_csr_genes",
        [](cuda_array_c<const int> index, cuda_array_c<const float> data,
           cuda_array_c<float> sums_genes, cuda_array_c<int> gene_ex, int nnz,
           std::uintptr_t stream) {
            launch_qc_csr_genes<float>(index.data(), data.data(),
                                       sums_genes.data(), gene_ex.data(), nnz,
                                       (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a,
        "nnz"_a, "stream"_a = 0);

    // sparse_qc_csr_genes - float64
    m.def(
        "sparse_qc_csr_genes",
        [](cuda_array_c<const int> index, cuda_array_c<const double> data,
           cuda_array_c<double> sums_genes, cuda_array_c<int> gene_ex, int nnz,
           std::uintptr_t stream) {
            launch_qc_csr_genes<double>(index.data(), data.data(),
                                        sums_genes.data(), gene_ex.data(), nnz,
                                        (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a,
        "nnz"_a, "stream"_a = 0);

    // sparse_qc_dense_cells - float32
    m.def(
        "sparse_qc_dense_cells",
        [](cuda_array_c<const float> data, cuda_array_c<float> sums_cells,
           cuda_array_c<int> cell_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_cells<float>(data.data(), sums_cells.data(),
                                         cell_ex.data(), n_cells, n_genes,
                                         (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "cell_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);

    // sparse_qc_dense_cells - float64
    m.def(
        "sparse_qc_dense_cells",
        [](cuda_array_c<const double> data, cuda_array_c<double> sums_cells,
           cuda_array_c<int> cell_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_cells<double>(data.data(), sums_cells.data(),
                                          cell_ex.data(), n_cells, n_genes,
                                          (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "cell_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);

    // sparse_qc_dense_genes - float32
    m.def(
        "sparse_qc_dense_genes",
        [](cuda_array_c<const float> data, cuda_array_c<float> sums_genes,
           cuda_array_c<int> gene_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_genes<float>(data.data(), sums_genes.data(),
                                         gene_ex.data(), n_cells, n_genes,
                                         (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);

    // sparse_qc_dense_genes - float64
    m.def(
        "sparse_qc_dense_genes",
        [](cuda_array_c<const double> data, cuda_array_c<double> sums_genes,
           cuda_array_c<int> gene_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_genes<double>(data.data(), sums_genes.data(),
                                          gene_ex.data(), n_cells, n_genes,
                                          (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);
}
