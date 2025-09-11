#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void sparse2dense_kernel(
    const int* __restrict__ indptr,
    const int* __restrict__ index,
    const T*   __restrict__ data,
    T*         __restrict__ out,
    long long major,
    long long minor,
    int c_switch)
{
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= major) return;
    long long start = (long long)indptr[row];
    long long stop  = (long long)indptr[row + 1];
    if (col >= (stop - start)) return;
    long long idx = (long long)index[start + col];
    if (idx >= minor) return;
    long long res_index = (c_switch == 1)
        ? (row * minor + idx)
        : (row + idx * major);
    atomicAdd(&out[res_index], data[start + col]);
}
