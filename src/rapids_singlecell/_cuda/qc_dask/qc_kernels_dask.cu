#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_qcd.cuh"

using namespace nb::literals;

constexpr int SPARSE_BLOCK_SIZE = 32;
constexpr int GENES_BLOCK_SIZE = 256;
constexpr int DENSE_BLOCK_DIM = 16;

template <typename T>
static inline void launch_qc_csr_cells(const int* indptr, const int* index,
                                       const T* data, T* sums_cells,
                                       int* cell_ex, int n_cells,
                                       cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_cells + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    qc_csr_cells_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, cell_ex, n_cells);
    CUDA_CHECK_LAST_ERROR(qc_csr_cells_kernel);
}

template <typename T>
static inline void launch_qc_csr_genes(const int* index, const T* data,
                                       T* sums_genes, int* gene_ex, int nnz,
                                       cudaStream_t stream) {
    int block = GENES_BLOCK_SIZE;
    int grid = (nnz + GENES_BLOCK_SIZE - 1) / GENES_BLOCK_SIZE;
    qc_csr_genes_kernel<T>
        <<<grid, block, 0, stream>>>(index, data, sums_genes, gene_ex, nnz);
    CUDA_CHECK_LAST_ERROR(qc_csr_genes_kernel);
}

template <typename T>
static inline void launch_qc_dense_cells(const T* data, T* sums_cells,
                                         int* cell_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_cells_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_cells, cell_ex, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_cells_kernel);
}

template <typename T>
static inline void launch_qc_dense_genes(const T* data, T* sums_genes,
                                         int* gene_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_genes_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_genes, gene_ex, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_genes_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // sparse_qc_csr_cells - float32
    m.def(
        "sparse_qc_csr_cells",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<int, Device> cell_ex, int n_cells,
           std::uintptr_t stream) {
            launch_qc_csr_cells<float>(indptr.data(), index.data(), data.data(),
                                       sums_cells.data(), cell_ex.data(),
                                       n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "cell_ex"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_csr_cells - float64
    m.def(
        "sparse_qc_csr_cells",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<int, Device> cell_ex, int n_cells,
           std::uintptr_t stream) {
            launch_qc_csr_cells<double>(
                indptr.data(), index.data(), data.data(), sums_cells.data(),
                cell_ex.data(), n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "cell_ex"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_csr_genes - float32
    m.def(
        "sparse_qc_csr_genes",
        [](gpu_array_c<const int, Device> index,
           gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, int nnz, std::uintptr_t stream) {
            launch_qc_csr_genes<float>(index.data(), data.data(),
                                       sums_genes.data(), gene_ex.data(), nnz,
                                       (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a,
        "nnz"_a, "stream"_a = 0);

    // sparse_qc_csr_genes - float64
    m.def(
        "sparse_qc_csr_genes",
        [](gpu_array_c<const int, Device> index,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, int nnz, std::uintptr_t stream) {
            launch_qc_csr_genes<double>(index.data(), data.data(),
                                        sums_genes.data(), gene_ex.data(), nnz,
                                        (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a,
        "nnz"_a, "stream"_a = 0);

    // sparse_qc_dense_cells - float32
    m.def(
        "sparse_qc_dense_cells",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<int, Device> cell_ex, int n_cells, int n_genes,
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
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<int, Device> cell_ex, int n_cells, int n_genes,
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
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, int n_cells, int n_genes,
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
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_genes<double>(data.data(), sums_genes.data(),
                                          gene_ex.data(), n_cells, n_genes,
                                          (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);
}

NB_MODULE(_qc_dask_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
