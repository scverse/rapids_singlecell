#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void colsum_kernel(const T* __restrict__ A, T* __restrict__ out, std::size_t rows,
                              std::size_t cols) {
  std::size_t tid = threadIdx.x;
  for (std::size_t col = blockIdx.x; col < cols; col += gridDim.x) {
    T acc = (T)0;
    for (std::size_t i = tid; i < rows; i += blockDim.x) {
      acc += A[i * cols + col];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    __shared__ T s[32];
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
      T val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : (T)0;
      for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
      if (threadIdx.x == 0) out[col] = val;
    }
  }
}

template <typename T>
__global__ void colsum_atomic_kernel(const T* __restrict__ A, T* __restrict__ out, std::size_t rows,
                                     std::size_t cols) {
  std::size_t tile_cols = (cols + 31) / 32;
  std::size_t tid = blockIdx.x;
  std::size_t tile_r = tid / tile_cols;
  std::size_t tile_c = tid % tile_cols;
  std::size_t row = tile_r * 32 + threadIdx.x;
  std::size_t col = tile_c * 32 + threadIdx.y;
  T v = (T)0;
  if (row < rows && col < cols) v = A[row * cols + col];
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  if (threadIdx.x == 0 && col < cols) atomicAdd(&out[col], v);
}
