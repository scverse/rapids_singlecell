#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_qc.cuh"

using namespace nb::literals;

constexpr int SPARSE_BLOCK_SIZE = 32;
constexpr int DENSE_BLOCK_DIM = 16;

template <typename T>
static inline void launch_qc_csc(const int* indptr, const int* index,
                                 const T* data, T* sums_cells, T* sums_genes,
                                 int* cell_ex, int* gene_ex, int n_genes,
                                 cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_genes + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    qc_csc_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, sums_genes, cell_ex, gene_ex, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_csc_kernel);
}

template <typename T>
static inline void launch_qc_csr(const int* indptr, const int* index,
                                 const T* data, T* sums_cells, T* sums_genes,
                                 int* cell_ex, int* gene_ex, int n_cells,
                                 cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_cells + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    qc_csr_kernel<T><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, sums_genes, cell_ex, gene_ex, n_cells);
    CUDA_CHECK_LAST_ERROR(qc_csr_kernel);
}

template <typename T>
static inline void launch_qc_dense(const T* data, T* sums_cells, T* sums_genes,
                                   int* cell_ex, int* gene_ex, int n_cells,
                                   int n_genes, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_cells, sums_genes, cell_ex, gene_ex, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_kernel);
}

template <typename T>
static inline void launch_qc_csc_sub(const int* indptr, const int* index,
                                     const T* data, T* sums_cells,
                                     const bool* mask, int n_genes,
                                     cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_genes + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    qc_csc_sub_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data,
                                                     sums_cells, mask, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_csc_sub_kernel);
}

template <typename T>
static inline void launch_qc_csr_sub(const int* indptr, const int* index,
                                     const T* data, T* sums_cells,
                                     const bool* mask, int n_cells,
                                     cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_cells + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    qc_csr_sub_kernel<T><<<grid, block, 0, stream>>>(indptr, index, data,
                                                     sums_cells, mask, n_cells);
    CUDA_CHECK_LAST_ERROR(qc_csr_sub_kernel);
}

template <typename T>
static inline void launch_qc_dense_sub(const T* data, T* sums_cells,
                                       const bool* mask, int n_cells,
                                       int n_genes, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_sub_kernel<T>
        <<<grid, block, 0, stream>>>(data, sums_cells, mask, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_sub_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // sparse_qc_csc - float32
    m.def(
        "sparse_qc_csc",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<float, Device> sums_genes,
           gpu_array_c<int, Device> cell_ex, gpu_array_c<int, Device> gene_ex,
           int n_genes, std::uintptr_t stream) {
            launch_qc_csc<float>(indptr.data(), index.data(), data.data(),
                                 sums_cells.data(), sums_genes.data(),
                                 cell_ex.data(), gene_ex.data(), n_genes,
                                 (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "cell_ex"_a, "gene_ex"_a, "n_genes"_a, "stream"_a = 0);

    // sparse_qc_csc - float64
    m.def(
        "sparse_qc_csc",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<double, Device> sums_genes,
           gpu_array_c<int, Device> cell_ex, gpu_array_c<int, Device> gene_ex,
           int n_genes, std::uintptr_t stream) {
            launch_qc_csc<double>(indptr.data(), index.data(), data.data(),
                                  sums_cells.data(), sums_genes.data(),
                                  cell_ex.data(), gene_ex.data(), n_genes,
                                  (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "cell_ex"_a, "gene_ex"_a, "n_genes"_a, "stream"_a = 0);

    // sparse_qc_csr - float32
    m.def(
        "sparse_qc_csr",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<float, Device> sums_genes,
           gpu_array_c<int, Device> cell_ex, gpu_array_c<int, Device> gene_ex,
           int n_cells, std::uintptr_t stream) {
            launch_qc_csr<float>(indptr.data(), index.data(), data.data(),
                                 sums_cells.data(), sums_genes.data(),
                                 cell_ex.data(), gene_ex.data(), n_cells,
                                 (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "cell_ex"_a, "gene_ex"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_csr - float64
    m.def(
        "sparse_qc_csr",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<double, Device> sums_genes,
           gpu_array_c<int, Device> cell_ex, gpu_array_c<int, Device> gene_ex,
           int n_cells, std::uintptr_t stream) {
            launch_qc_csr<double>(indptr.data(), index.data(), data.data(),
                                  sums_cells.data(), sums_genes.data(),
                                  cell_ex.data(), gene_ex.data(), n_cells,
                                  (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "cell_ex"_a, "gene_ex"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_dense - float32
    m.def(
        "sparse_qc_dense",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<float, Device> sums_genes,
           gpu_array_c<int, Device> cell_ex, gpu_array_c<int, Device> gene_ex,
           int n_cells, int n_genes, std::uintptr_t stream) {
            launch_qc_dense<float>(data.data(), sums_cells.data(),
                                   sums_genes.data(), cell_ex.data(),
                                   gene_ex.data(), n_cells, n_genes,
                                   (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "sums_genes"_a, "cell_ex"_a,
        "gene_ex"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);

    // sparse_qc_dense - float64
    m.def(
        "sparse_qc_dense",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<double, Device> sums_genes,
           gpu_array_c<int, Device> cell_ex, gpu_array_c<int, Device> gene_ex,
           int n_cells, int n_genes, std::uintptr_t stream) {
            launch_qc_dense<double>(data.data(), sums_cells.data(),
                                    sums_genes.data(), cell_ex.data(),
                                    gene_ex.data(), n_cells, n_genes,
                                    (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "sums_genes"_a, "cell_ex"_a,
        "gene_ex"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);

    // sparse_qc_csc_sub - float32
    m.def(
        "sparse_qc_csc_sub",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_genes,
           std::uintptr_t stream) {
            launch_qc_csc_sub<float>(indptr.data(), index.data(), data.data(),
                                     sums_cells.data(), mask.data(), n_genes,
                                     (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "mask"_a, "n_genes"_a, "stream"_a = 0);

    // sparse_qc_csc_sub - float64
    m.def(
        "sparse_qc_csc_sub",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_genes,
           std::uintptr_t stream) {
            launch_qc_csc_sub<double>(indptr.data(), index.data(), data.data(),
                                      sums_cells.data(), mask.data(), n_genes,
                                      (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "mask"_a, "n_genes"_a, "stream"_a = 0);

    // sparse_qc_csr_sub - float32
    m.def(
        "sparse_qc_csr_sub",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_cells,
           std::uintptr_t stream) {
            launch_qc_csr_sub<float>(indptr.data(), index.data(), data.data(),
                                     sums_cells.data(), mask.data(), n_cells,
                                     (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "mask"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_csr_sub - float64
    m.def(
        "sparse_qc_csr_sub",
        [](gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> index,
           gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_cells,
           std::uintptr_t stream) {
            launch_qc_csr_sub<double>(indptr.data(), index.data(), data.data(),
                                      sums_cells.data(), mask.data(), n_cells,
                                      (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "mask"_a, "n_cells"_a, "stream"_a = 0);

    // sparse_qc_dense_sub - float32
    m.def(
        "sparse_qc_dense_sub",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_sub<float>(data.data(), sums_cells.data(),
                                       mask.data(), n_cells, n_genes,
                                       (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "mask"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);

    // sparse_qc_dense_sub - float64
    m.def(
        "sparse_qc_dense_sub",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<double, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_sub<double>(data.data(), sums_cells.data(),
                                        mask.data(), n_cells, n_genes,
                                        (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "mask"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);
}

NB_MODULE(_qc_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
