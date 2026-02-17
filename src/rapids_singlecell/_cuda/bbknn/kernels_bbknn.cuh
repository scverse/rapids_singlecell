#pragma once

#include <cuda_runtime.h>

__global__ void find_top_k_per_row_kernel(const float* __restrict__ data,
                                          const int* __restrict__ indptr, const int n_rows,
                                          const int trim, float* __restrict__ vals) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }

  int start = indptr[row];
  int end = indptr[row + 1];
  int length = end - start;

  if (length <= trim) {
    vals[row] = 0.0f;  // insufficient elements
    return;
  }

  extern __shared__ float shared_memory[];
  int shared_offset = threadIdx.x * trim;
  float* top_k = &shared_memory[shared_offset];

  // Initialize top_k with zeros
  for (int i = 0; i < trim; ++i) {
    top_k[i] = 0.0f;
  }

  int min_index = 0;
  // Process each element in the row
  for (int idx = start; idx < end; ++idx) {
    float v = data[idx];
    if (v <= top_k[min_index]) {
      continue;
    }
    // Replace the current minimum in top_k
    top_k[min_index] = v;
    // Find new smallest element index in top_k
    for (int i = 0; i < trim; ++i) {
      if (top_k[i] < top_k[min_index]) {
        min_index = i;
      }
    }
  }

  vals[row] = top_k[min_index];
}

__global__ void cut_smaller_kernel(const int* __restrict__ indptr, const int* __restrict__ index,
                                   float* __restrict__ data, const float* __restrict__ vals,
                                   const int n_rows) {
  int row_id = blockIdx.x;
  if (row_id >= n_rows) {
    return;
  }

  int start_idx = indptr[row_id];
  int stop_idx = indptr[row_id + 1];
  float cut_row = vals[row_id];

  for (int i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
    float neighbor_cut = vals[index[i]];
    float cut = fmaxf(neighbor_cut, cut_row);
    if (data[i] < cut) {
      data[i] = 0.0f;
    }
  }
}
