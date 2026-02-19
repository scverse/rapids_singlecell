#pragma once

#include <cuda_runtime.h>

// All scale kernels assume std[col] > 0, guaranteed by the Python caller
// (_scale.py clips std to a minimum value before invoking these kernels).

template <typename T>
__global__ void csc_scale_diff_kernel(const int* __restrict__ indptr,
                                      T* __restrict__ data,
                                      const T* __restrict__ std, int ncols) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    int start_idx = indptr[col];
    int stop_idx = indptr[col + 1];
    T diver = T(1) / std[col];
    for (int i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
        data[i] *= diver;
    }
}

template <typename T>
__global__ void csr_scale_diff_kernel(const int* __restrict__ indptr,
                                      const int* __restrict__ indices,
                                      T* __restrict__ data,
                                      const T* __restrict__ std,
                                      const int* __restrict__ mask, T clipper,
                                      int nrows) {
    int row = blockIdx.x;
    if (row >= nrows) return;
    if (mask[row]) {
        int start_idx = indptr[row];
        int stop_idx = indptr[row + 1];
        for (int i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
            int idx = indices[i];
            T res = data[i] / std[idx];
            data[i] = res < clipper ? res : clipper;
        }
    }
}

template <typename T>
__global__ void dense_scale_center_diff_kernel(
    T* data, const T* __restrict__ mean, const T* __restrict__ std,
    const int* __restrict__ mask, T clipper, long long nrows, long long ncols) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (row < nrows && col < ncols) {
        if (mask[row]) {
            T res = data[row * ncols + col] - mean[col];
            res = res / std[col];
            if (res > clipper) res = clipper;
            if (res < -clipper) res = -clipper;
            data[row * ncols + col] = res;
        }
    }
}

template <typename T>
__global__ void dense_scale_diff_kernel(T* __restrict__ data,
                                        const T* __restrict__ std,
                                        const int* __restrict__ mask, T clipper,
                                        long long nrows, long long ncols) {
    long long row = (long long)(blockIdx.x * blockDim.x + threadIdx.x);
    long long col = (long long)(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < nrows && col < ncols) {
        if (mask[row]) {
            T res = data[row * ncols + col] / std[col];
            data[row * ncols + col] = res < clipper ? res : clipper;
        }
    }
}
