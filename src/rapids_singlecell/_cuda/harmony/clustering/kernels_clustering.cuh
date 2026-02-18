#pragma once

#include <cuda_runtime.h>
#include <type_traits>

// ---- Fused entropy kernel ----
// One block per row. Row-normalize R, accumulate x*log(x+eps), atomicAdd scaled result.
template <typename T>
__global__ void entropy_kernel(const T* __restrict__ R, T sigma, int n_cells, int n_clusters,
                               T* __restrict__ obj_out) {
  int row = blockIdx.x;
  if (row >= n_cells) return;

  const T* R_row = R + (size_t)row * n_clusters;

  // Phase 1: row sum
  T row_sum = T(0);
  for (int col = threadIdx.x; col < n_clusters; col += blockDim.x) row_sum += R_row[col];

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);

  __shared__ T shared[32];
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int num_warps = (blockDim.x + 31) >> 5;

  if (lane == 0) shared[warp_id] = row_sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    T val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x == 0) shared[0] = val;
  }
  __syncthreads();
  T inv_rsum = T(1) / shared[0];

  // Phase 2: entropy = sum(x_norm * log(x_norm + eps))
  T entropy = T(0);
  for (int col = threadIdx.x; col < n_clusters; col += blockDim.x) {
    T x = R_row[col] * inv_rsum;
    if constexpr (std::is_same<T, float>::value)
      entropy += x * logf(x + T(1e-12));
    else
      entropy += x * log(x + T(1e-12));
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    entropy += __shfl_down_sync(0xffffffff, entropy, offset);

  if (lane == 0) shared[warp_id] = entropy;
  __syncthreads();
  if (threadIdx.x < 32) {
    T val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x == 0) atomicAdd(obj_out, sigma * val);
  }
}

// ---- Diversity kernel ----
// sigma * sum_{b,k} theta[b] * O[b,k] * log((O[b,k]+1)/(E[b,k]+1))
template <typename T>
__global__ void diversity_kernel(const T* __restrict__ O, const T* __restrict__ E,
                                 const T* __restrict__ theta, T sigma, int n_batches,
                                 int n_clusters, T* __restrict__ obj_out) {
  T acc = T(0);
  int total = n_batches * n_clusters;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
    int batch = i / n_clusters;
    T ratio = (O[i] + T(1)) / (E[i] + T(1));
    T log_val;
    if constexpr (std::is_same<T, float>::value)
      log_val = logf(ratio);
    else
      log_val = log(ratio);
    acc += theta[batch] * O[i] * log_val;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xffffffff, acc, offset);

  __shared__ T shared[32];
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int num_warps = (blockDim.x + 31) >> 5;

  if (lane == 0) shared[warp_id] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    T val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x == 0) atomicAdd(obj_out, sigma * val);
  }
}

// ---- PCG hash for random shuffle keys ----
__global__ void pcg_hash_kernel(unsigned int* __restrict__ out, int n, unsigned int seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    unsigned int state = static_cast<unsigned int>(i) ^ seed;
    state = state * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    out[i] = (word >> 22u) ^ word;
  }
}

// ---- Iota: out[i] = i ----
__global__ void iota_kernel(int* __restrict__ out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    out[i] = i;
}
