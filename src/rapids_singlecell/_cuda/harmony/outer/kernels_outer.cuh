#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void outer_kernel(T* __restrict__ E, const T* __restrict__ Pr_b,
                             const T* __restrict__ R_sum, long long n_cats, long long n_pcs,
                             long long switcher) {
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
__global__ void harmony_correction_kernel(T* __restrict__ Z, const T* __restrict__ W,
                                          const int* __restrict__ cats, const T* __restrict__ R,
                                          long long n_cells, long long n_pcs) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells * n_pcs) return;
  long long cell_idx = i / n_pcs;
  long long pc_idx = i % n_pcs;
  int cat = cats[cell_idx];
  T correction = W[(cat + 1) * n_pcs + pc_idx] * R[cell_idx];
  Z[i] -= correction;
}
