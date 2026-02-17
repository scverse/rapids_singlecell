#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void mean_var_major_kernel(const int* __restrict__ indptr,
                                      const int* __restrict__ indices, const T* __restrict__ data,
                                      double* __restrict__ means, double* __restrict__ vars,
                                      int major, int /*minor*/) {
  int major_idx = blockIdx.x;
  if (major_idx >= major) return;

  int start_idx = indptr[major_idx];
  int stop_idx = indptr[major_idx + 1];

  __shared__ double mean_place[64];
  __shared__ double var_place[64];

  mean_place[threadIdx.x] = 0.0;
  var_place[threadIdx.x] = 0.0;
  __syncthreads();

  for (int minor_idx = start_idx + threadIdx.x; minor_idx < stop_idx; minor_idx += blockDim.x) {
    double value = static_cast<double>(data[minor_idx]);
    mean_place[threadIdx.x] += value;
    var_place[threadIdx.x] += value * value;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
      var_place[threadIdx.x] += var_place[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    means[major_idx] = mean_place[0];
    vars[major_idx] = var_place[0];
  }
}

template <typename T>
__global__ void mean_var_minor_kernel(const int* __restrict__ indices, const T* __restrict__ data,
                                      double* __restrict__ means, double* __restrict__ vars,
                                      int nnz) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= nnz) return;
  double value = static_cast<double>(data[idx]);
  int minor_pos = indices[idx];
  atomicAdd(&means[minor_pos], value);
  atomicAdd(&vars[minor_pos], value * value);
}
