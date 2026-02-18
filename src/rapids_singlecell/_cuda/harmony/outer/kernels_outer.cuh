#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void outer_kernel(T* __restrict__ E, const T* __restrict__ Pr_b,
                             const T* __restrict__ R_sum, long long n_cats,
                             long long n_pcs, long long switcher) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    long long N = n_cats * n_pcs;
    if (i >= N) return;
    long long row = i / n_pcs;
    long long col = i % n_pcs;
    if (switcher == 0)
        E[i] -= (Pr_b[row] * R_sum[col]);
    else
        E[i] += (Pr_b[row] * R_sum[col]);
}

template <typename T>
__global__ void harmony_correction_kernel(T* __restrict__ Z,
                                          const T* __restrict__ W,
                                          const int* __restrict__ cats,
                                          const T* __restrict__ R,
                                          long long n_cells, long long n_pcs) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * n_pcs) return;
    long long cell_idx = i / n_pcs;
    long long pc_idx = i % n_pcs;
    int cat = cats[cell_idx];
    T correction = W[(cat + 1) * n_pcs + pc_idx] * R[cell_idx];
    Z[i] -= correction;
}

// ---------- batched_correction ----------
// Each thread handles one (cell, pc) pair, accumulating corrections from all
// clusters. W_all layout: (n_clusters, n_batches+1, n_pcs) row-major
template <typename T>
__global__ void batched_correction_kernel(T* __restrict__ Z,
                                          const T* __restrict__ W_all,
                                          const int* __restrict__ cats,
                                          const T* __restrict__ R, int n_cells,
                                          int n_pcs, int n_clusters,
                                          int n_batches_p1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells * n_pcs) return;

    int cell = idx / n_pcs;
    int pc = idx % n_pcs;
    int cat = cats[cell];

    T total_correction = T(0);
    for (int k = 0; k < n_clusters; k++) {
        T w_val =
            W_all[(long long)k * n_batches_p1 * n_pcs + (cat + 1) * n_pcs + pc];
        T r_val = R[cell * n_clusters + k];
        total_correction += w_val * r_val;
    }

    Z[idx] -= total_correction;
}
