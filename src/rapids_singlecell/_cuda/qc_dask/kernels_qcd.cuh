#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void qc_csr_cells_kernel(const int* __restrict__ indptr,
                                    const int* __restrict__ index,
                                    const T* __restrict__ data,
                                    T* __restrict__ sums_cells,
                                    int* __restrict__ cell_ex, int n_cells) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= n_cells) return;
    int start_idx = indptr[cell];
    int stop_idx = indptr[cell + 1];
    T sums = T(0);
    int ex = 0;
    for (int p = start_idx; p < stop_idx; ++p) {
        sums += data[p];
        ++ex;
    }
    sums_cells[cell] = sums;
    cell_ex[cell] = ex;
}

template <typename T>
__global__ void qc_csr_genes_kernel(const int* __restrict__ index,
                                    const T* __restrict__ data,
                                    T* __restrict__ sums_genes,
                                    int* __restrict__ gene_ex, int nnz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nnz) return;
    int g = index[i];
    T v = data[i];
    atomicAdd(&sums_genes[g], v);
    atomicAdd(&gene_ex[g], 1);
}

template <typename T>
__global__ void qc_dense_cells_kernel(const T* __restrict__ data,
                                      T* __restrict__ sums_cells,
                                      int* __restrict__ cell_ex, int n_cells,
                                      int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) return;
    long long idx = (long long)cell * n_genes + gene;
    T v = data[idx];
    if (v > T(0)) {
        atomicAdd(&sums_cells[cell], v);
        atomicAdd(&cell_ex[cell], 1);
    }
}

template <typename T>
__global__ void qc_dense_genes_kernel(const T* __restrict__ data,
                                      T* __restrict__ sums_genes,
                                      int* __restrict__ gene_ex, int n_cells,
                                      int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) return;
    long long idx = (long long)cell * n_genes + gene;
    T v = data[idx];
    if (v > T(0)) {
        atomicAdd(&sums_genes[gene], v);
        atomicAdd(&gene_ex[gene], 1);
    }
}
