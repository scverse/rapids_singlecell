#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void normalize_kernel_optimized(T* X, long long rows, long long cols) {
  __shared__ T shared[32];
  long long row = blockIdx.x;
  long long tid = threadIdx.x;
  if (row >= rows) return;
  T norm = (T)0;
  for (long long col = tid; col < cols; col += blockDim.x) {
    T v = X[row * cols + col];
    norm += (v < 0 ? -v : v);
  }
  shared[tid] = norm;
  __syncthreads();
  for (long long offset = 16; offset > 0; offset /= 2) {
    shared[tid] += __shfl_down_sync(0xFFFFFFFF, shared[tid], offset);
  }
  __syncthreads();
  if (tid == 0) {
    T final_norm = shared[0];
    final_norm = final_norm < (T)1e-12 ? (T)1e-12 : final_norm;
    shared[0] = (T)1 / final_norm;
  }
  __syncthreads();
  for (long long col = tid; col < cols; col += blockDim.x) {
    X[row * cols + col] *= shared[0];
  }
}
