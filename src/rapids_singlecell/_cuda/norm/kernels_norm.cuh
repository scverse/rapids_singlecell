#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void dense_row_scale_kernel(T* __restrict__ data, int nrows, int ncols, T target_sum) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) {
    return;
  }

  T sum = (T)0;
  int base = row * ncols;
  for (int c = 0; c < ncols; ++c) {
    sum += data[base + c];
  }
  if (sum > (T)0) {
    T scale = target_sum / sum;
    for (int c = 0; c < ncols; ++c) {
      data[base + c] *= scale;
    }
  }
}

template <typename T>
__global__ void csr_row_scale_kernel(const int* __restrict__ indptr, T* __restrict__ data,
                                     int nrows, T target_sum) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) {
    return;
  }
  int start = indptr[row];
  int stop = indptr[row + 1];
  T sum = (T)0;
  for (int i = start; i < stop; ++i) {
    sum += data[i];
  }
  if (sum > (T)0) {
    T scale = target_sum / sum;
    for (int i = start; i < stop; ++i) {
      data[i] *= scale;
    }
  }
}

template <typename T>
__global__ void csr_sum_major_kernel(const int* __restrict__ indptr, const T* __restrict__ data,
                                     T* __restrict__ sums, int major) {
  int major_idx = blockIdx.x;
  if (major_idx >= major) {
    return;
  }
  extern __shared__ unsigned char smem[];
  T* sum_place = reinterpret_cast<T*>(smem);

  // initialize
  sum_place[threadIdx.x] = (T)0;
  __syncthreads();

  int start = indptr[major_idx];
  int stop = indptr[major_idx + 1];
  for (int minor_idx = start + threadIdx.x; minor_idx < stop; minor_idx += blockDim.x) {
    sum_place[threadIdx.x] += data[minor_idx];
  }
  __syncthreads();

  // reduction in shared memory
  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sum_place[threadIdx.x] += sum_place[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sums[major_idx] = sum_place[0];
  }
}
