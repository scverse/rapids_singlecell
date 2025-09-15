#pragma once

#include <cuda_runtime.h>

template <typename T, bool C_ORDER>
__global__ void sparse2dense_kernel(const int* __restrict__ indptr, const int* __restrict__ index,
                                    const T* __restrict__ data, T* __restrict__ out,
                                    long long major, long long minor) {
  long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= major) return;

  long long start = (long long)indptr[row];
  long long stop = (long long)indptr[row + 1];
  long long nnz_in_row = stop - start;
  if (col >= nnz_in_row) return;

  long long j = (long long)index[start + col];
  if (j >= minor) return;

  long long res_index;
  if constexpr (C_ORDER) {
    // row-major: [row, j] -> row*minor + j
    res_index = row * minor + j;
  } else {
    // col-major (Fortran): [row, j] -> row + j*major
    res_index = row + j * major;
  }

  // If duplicates per row/col are impossible, replace with a simple store.
  atomicAdd(&out[res_index], data[start + col]);
}
