#pragma once

#include <cuda_runtime.h>
#include <type_traits>

template <typename T>
__global__ void kmeans_err_kernel(const T* __restrict__ r, const T* __restrict__ dot, std::size_t n,
                                  T* __restrict__ out) {
  T acc = (T)0;
  using Vec = typename std::conditional<std::is_same<T, float>::value, float4, double4>::type;

  std::size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const std::size_t stride = gridDim.x * blockDim.x * 4;

  while (i + 3 < n) {
    Vec r4 = *(const Vec*)(r + i);
    Vec dot4 = *(const Vec*)(dot + i);
    acc += ((T*)&r4)[0] * (T)2 * ((T)1 - ((T*)&dot4)[0]);
    acc += ((T*)&r4)[1] * (T)2 * ((T)1 - ((T*)&dot4)[1]);
    acc += ((T*)&r4)[2] * (T)2 * ((T)1 - ((T*)&dot4)[2]);
    acc += ((T*)&r4)[3] * (T)2 * ((T)1 - ((T*)&dot4)[3]);
    i += stride;
  }
  while (i < n) {
    T rv = r[i];
    T dotv = dot[i];
    acc += rv * (T)2 * ((T)1 - dotv);
    i++;
  }

  for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xffffffff, acc, offset);
  __shared__ T s[32];
  if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    T val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : (T)0;
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x == 0) atomicAdd(out, val);
  }
}
