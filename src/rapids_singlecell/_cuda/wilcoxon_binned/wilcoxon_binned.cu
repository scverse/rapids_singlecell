#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_wilcoxon_binned.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_dense_hist(const T* X, const int* gcodes,
                                     unsigned int* hist, int n_cells,
                                     int n_genes, int n_groups, int n_bins,
                                     double bin_low, double inv_bin_width,
                                     cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(n_genes);
    dense_hist_kernel<T><<<grid, block, 0, stream>>>(X, gcodes, hist, n_cells,
                                                     n_genes, n_groups, n_bins,
                                                     bin_low, inv_bin_width);
}

template <typename T>
static inline void launch_csr_hist(const T* data, const int* indices,
                                   const int* indptr, const int* gcodes,
                                   unsigned int* hist, int n_cells, int n_genes,
                                   int n_groups, int n_bins, double bin_low,
                                   double inv_bin_width, int gene_start,
                                   cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(n_cells);
    csr_hist_kernel<T><<<grid, block, 0, stream>>>(
        data, indices, indptr, gcodes, hist, n_cells, n_genes, n_groups, n_bins,
        bin_low, inv_bin_width, gene_start);
}

template <typename T>
static inline void launch_csc_hist(const T* data, const int* indices,
                                   const int* indptr, const int* gcodes,
                                   unsigned int* hist, int n_cells, int n_genes,
                                   int n_groups, int n_bins, double bin_low,
                                   double inv_bin_width, int gene_start,
                                   cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(n_genes);
    csc_hist_kernel<T><<<grid, block, 0, stream>>>(
        data, indices, indptr, gcodes, hist, n_cells, n_genes, n_groups, n_bins,
        bin_low, inv_bin_width, gene_start);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // dense_hist - float32
    m.def(
        "dense_hist",
        [](gpu_array_f<const float, Device> X,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           std::uintptr_t stream) {
            launch_dense_hist<float>(
                X.data(), gcodes.data(), hist.data(), n_cells, n_genes,
                n_groups, n_bins, bin_low, inv_bin_width, (cudaStream_t)stream);
        },
        "X"_a, "gcodes"_a, "hist"_a, nb::kw_only(), "n_cells"_a, "n_genes"_a,
        "n_groups"_a, "n_bins"_a, "bin_low"_a, "inv_bin_width"_a,
        "stream"_a = 0);

    // dense_hist - float64
    m.def(
        "dense_hist",
        [](gpu_array_f<const double, Device> X,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           std::uintptr_t stream) {
            launch_dense_hist<double>(
                X.data(), gcodes.data(), hist.data(), n_cells, n_genes,
                n_groups, n_bins, bin_low, inv_bin_width, (cudaStream_t)stream);
        },
        "X"_a, "gcodes"_a, "hist"_a, nb::kw_only(), "n_cells"_a, "n_genes"_a,
        "n_groups"_a, "n_bins"_a, "bin_low"_a, "inv_bin_width"_a,
        "stream"_a = 0);

    // csr_hist - float32
    m.def(
        "csr_hist",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           int gene_start, std::uintptr_t stream) {
            launch_csr_hist<float>(data.data(), indices.data(), indptr.data(),
                                   gcodes.data(), hist.data(), n_cells, n_genes,
                                   n_groups, n_bins, bin_low, inv_bin_width,
                                   gene_start, (cudaStream_t)stream);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "n_cells"_a, "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "gene_start"_a, "stream"_a = 0);

    // csr_hist - float64
    m.def(
        "csr_hist",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           int gene_start, std::uintptr_t stream) {
            launch_csr_hist<double>(
                data.data(), indices.data(), indptr.data(), gcodes.data(),
                hist.data(), n_cells, n_genes, n_groups, n_bins, bin_low,
                inv_bin_width, gene_start, (cudaStream_t)stream);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "n_cells"_a, "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "gene_start"_a, "stream"_a = 0);

    // csc_hist - float32
    m.def(
        "csc_hist",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           int gene_start, std::uintptr_t stream) {
            launch_csc_hist<float>(data.data(), indices.data(), indptr.data(),
                                   gcodes.data(), hist.data(), n_cells, n_genes,
                                   n_groups, n_bins, bin_low, inv_bin_width,
                                   gene_start, (cudaStream_t)stream);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "n_cells"_a, "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "gene_start"_a, "stream"_a = 0);

    // csc_hist - float64
    m.def(
        "csc_hist",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           int gene_start, std::uintptr_t stream) {
            launch_csc_hist<double>(
                data.data(), indices.data(), indptr.data(), gcodes.data(),
                hist.data(), n_cells, n_genes, n_groups, n_bins, bin_low,
                inv_bin_width, gene_start, (cudaStream_t)stream);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "n_cells"_a, "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "gene_start"_a, "stream"_a = 0);
}

NB_MODULE(_wilcoxon_binned_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
