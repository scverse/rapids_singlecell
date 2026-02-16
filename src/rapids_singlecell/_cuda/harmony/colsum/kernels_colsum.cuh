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

// Block handles one column-tile, processes rows_per_tile rows.
// blockIdx.x = column tile, blockIdx.y = row tile.
// Uses shared memory to reduce atomics: one per column per block.
template <typename T>
__global__ void colsum_atomic_kernel(const T* __restrict__ A, T* __restrict__ out, std::size_t rows,
                                     std::size_t cols, std::size_t rows_per_tile) {
  __shared__ T col_sums[32];

  std::size_t col = blockIdx.x * 32 + threadIdx.y;
  std::size_t start_row = blockIdx.y * rows_per_tile;
  std::size_t end_row = start_row + rows_per_tile;
  if (end_row > rows) end_row = rows;

  // Initialize shared memory
  if (threadIdx.x < 32) col_sums[threadIdx.x] = (T)0;
  __syncthreads();

  // Each thread accumulates multiple rows
  T acc = (T)0;
  if (col < cols) {
    for (std::size_t row = start_row + threadIdx.x; row < end_row; row += 32) {
      acc += A[row * cols + col];
    }
  }

  // Warp-level reduction across 32 threads (different rows)
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);

  // Lane 0 of each warp accumulates into shared memory
  if (threadIdx.x == 0 && threadIdx.y < 32) atomicAdd(&col_sums[threadIdx.y], acc);
  __syncthreads();

  // First warp writes to global memory (one atomic per column)
  if (threadIdx.x == 0 && threadIdx.y < 32) {
    std::size_t out_col = blockIdx.x * 32 + threadIdx.y;
    if (out_col < cols && col_sums[threadIdx.y] != (T)0)
      atomicAdd(&out[out_col], col_sums[threadIdx.y]);
  }
}
