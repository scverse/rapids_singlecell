#pragma once

#include <cuda_runtime.h>

// sparse -> dense aggregate (CSR by cells), mask per cell, cats per cell
template <typename T>
__global__ void csr_aggr_kernel(const int* __restrict__ indptr,
                                const int* __restrict__ index,
                                const T* __restrict__ data,
                                double* __restrict__ out,
                                const int* __restrict__ cats,
                                const bool* __restrict__ mask, size_t n_cells,
                                size_t n_genes, size_t n_groups) {
    size_t cell = blockIdx.x;
    if (cell >= n_cells || !mask[cell]) return;
    int cell_start = indptr[cell];
    int cell_end = indptr[cell + 1];
    size_t group = static_cast<size_t>(cats[cell]);
    for (int p = cell_start + threadIdx.x; p < cell_end; p += blockDim.x) {
        size_t gene_pos = static_cast<size_t>(index[p]);
        double v = static_cast<double>(data[p]);
        atomicAdd(&out[group * n_genes + gene_pos], v);
        atomicAdd(&out[group * n_genes + gene_pos + n_genes * n_groups], 1.0);
        atomicAdd(&out[group * n_genes + gene_pos + 2 * n_genes * n_groups],
                  v * v);
    }
}

// sparse -> dense aggregate (CSC by genes), mask per cell, cats per cell
template <typename T>
__global__ void csc_aggr_kernel(const int* __restrict__ indptr,
                                const int* __restrict__ index,
                                const T* __restrict__ data,
                                double* __restrict__ out,
                                const int* __restrict__ cats,
                                const bool* __restrict__ mask, size_t n_cells,
                                size_t n_genes, size_t n_groups) {
    size_t gene = blockIdx.x;
    if (gene >= n_genes) return;
    int gene_start = indptr[gene];
    int gene_end = indptr[gene + 1];
    for (int p = gene_start + threadIdx.x; p < gene_end; p += blockDim.x) {
        size_t cell = static_cast<size_t>(index[p]);
        if (!mask[cell]) continue;
        size_t group = static_cast<size_t>(cats[cell]);
        double v = static_cast<double>(data[p]);
        atomicAdd(&out[group * n_genes + gene], v);
        atomicAdd(&out[group * n_genes + gene + n_genes * n_groups], 1.0);
        atomicAdd(&out[group * n_genes + gene + 2 * n_genes * n_groups], v * v);
    }
}

// sparse -> sparse copy (CSR by cells) row/col/value from one to another by
// cats/mask
template <typename T>
__global__ void csr_to_coo_kernel(const int* __restrict__ indptr,
                                  const int* __restrict__ index,
                                  const T* __restrict__ data,
                                  int* __restrict__ row, int* __restrict__ col,
                                  double* __restrict__ ndata,
                                  const int* __restrict__ cats,
                                  const bool* __restrict__ mask, int n_cells) {
    int cell = blockIdx.x;
    if (cell >= n_cells || !mask[cell]) return;
    int start = indptr[cell];
    int end = indptr[cell + 1];
    int group = cats[cell];
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        int g = index[p];
        ndata[p] = static_cast<double>(data[p]);
        row[p] = group;
        col[p] = g;
    }
}

// variance adjust per group (CSR-like segment)
__global__ void sparse_var_kernel(const int* __restrict__ indptr,
                                  const int* __restrict__ index,
                                  double* __restrict__ data,
                                  const double* __restrict__ mean_data,
                                  double* __restrict__ n_cells, int dof,
                                  int n_groups) {
    int group = blockIdx.x;
    if (group >= n_groups) return;
    int start = indptr[group];
    int end = indptr[group + 1];
    double doffer =
        n_cells[group] / (n_cells[group] - static_cast<double>(dof));
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        double var = data[p];
        double mean_sq = mean_data[p] * mean_data[p];
        var = var - mean_sq;
        data[p] = var * doffer;
    }
}

// dense C-order aggregator
template <typename T>
__global__ void dense_aggr_kernel_C(const T* __restrict__ data,
                                    double* __restrict__ out,
                                    const int* __restrict__ cats,
                                    const bool* __restrict__ mask,
                                    size_t n_cells, size_t n_genes,
                                    size_t n_groups) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t N = n_cells * n_genes;
    while (i < N) {
        size_t cell = i / n_genes;
        size_t gene = i % n_genes;
        if (mask[cell]) {
            size_t group = static_cast<size_t>(cats[cell]);
            double v = static_cast<double>(data[cell * n_genes + gene]);
            if (v != 0.0) {
                atomicAdd(&out[group * n_genes + gene], v);
                atomicAdd(&out[group * n_genes + gene + n_genes * n_groups],
                          1.0);
                atomicAdd(&out[group * n_genes + gene + 2 * n_genes * n_groups],
                          v * v);
            }
        }
        i += stride;
    }
}

// dense F-order aggregator
template <typename T>
__global__ void dense_aggr_kernel_F(const T* __restrict__ data,
                                    double* __restrict__ out,
                                    const int* __restrict__ cats,
                                    const bool* __restrict__ mask,
                                    size_t n_cells, size_t n_genes,
                                    size_t n_groups) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t N = n_cells * n_genes;
    while (i < N) {
        size_t cell = i % n_cells;
        size_t gene = i / n_cells;
        if (mask[cell]) {
            size_t group = static_cast<size_t>(cats[cell]);
            double v = static_cast<double>(data[gene * n_cells + cell]);
            if (v != 0.0) {
                atomicAdd(&out[group * n_genes + gene], v);
                atomicAdd(&out[group * n_genes + gene + n_genes * n_groups],
                          1.0);
                atomicAdd(&out[group * n_genes + gene + 2 * n_genes * n_groups],
                          v * v);
            }
        }
        i += stride;
    }
}
