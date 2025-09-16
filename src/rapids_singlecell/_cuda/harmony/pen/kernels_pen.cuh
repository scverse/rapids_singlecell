#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void pen_kernel(T* __restrict__ R, const T* __restrict__ penalty,
                           const int* __restrict__ cats, std::size_t n_rows, std::size_t n_cols) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t N = n_rows * n_cols;
  if (i >= N) return;
  std::size_t row = i / n_cols;
  std::size_t col = i % n_cols;
  int cat = cats[row];
  T scale = penalty[(std::size_t)cat * n_cols + col];
  R[i] *= scale;
}
