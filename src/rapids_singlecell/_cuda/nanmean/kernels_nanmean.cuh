#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void nan_mean_minor_kernel(const int* __restrict__ index,
                                      const T* __restrict__ data,
                                      double* __restrict__ means,
                                      int* __restrict__ nans,
                                      const bool* __restrict__ mask, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) {
        return;
    }
    int minor_pos = index[idx];
    if (mask[minor_pos] == false) {
        return;
    }
    T v = data[idx];
    if (isnan((double)v)) {
        atomicAdd(&nans[minor_pos], 1);
    } else {
        atomicAdd(&means[minor_pos], (double)v);
    }
}

template <typename T>
__global__ void nan_mean_major_kernel(const int* __restrict__ indptr,
                                      const int* __restrict__ index,
                                      const T* __restrict__ data,
                                      double* __restrict__ means,
                                      int* __restrict__ nans,
                                      const bool* __restrict__ mask, int major,
                                      int minor) {
    int major_idx = blockIdx.x;
    if (major_idx >= major) {
        return;
    }
    int start_idx = indptr[major_idx];
    int stop_idx = indptr[major_idx + 1];

    __shared__ double mean_place[64];
    __shared__ int nan_place[64];

    mean_place[threadIdx.x] = 0.0;
    nan_place[threadIdx.x] = 0;
    __syncthreads();

    for (int minor_idx = start_idx + threadIdx.x; minor_idx < stop_idx;
         minor_idx += blockDim.x) {
        int gene_number = index[minor_idx];
        if (mask[gene_number]) {
            T v = data[minor_idx];
            if (isnan((double)v)) {
                nan_place[threadIdx.x] += 1;
            } else {
                mean_place[threadIdx.x] += (double)v;
            }
        }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
            nan_place[threadIdx.x] += nan_place[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        means[major_idx] = mean_place[0];
        nans[major_idx] = nan_place[0];
    }
}
