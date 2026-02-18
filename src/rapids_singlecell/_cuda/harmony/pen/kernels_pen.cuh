#pragma once

#include <cuda_runtime.h>

// One block per row (cell). Computes exp(term*(1-sim)) * penalty, then row-normalizes.
template <typename T>
__global__ void fused_pen_norm_kernel(const T* __restrict__ similarities,
                                      const T* __restrict__ penalty, const int* __restrict__ cats,
                                      const size_t* __restrict__ idx_in, T* __restrict__ R_out,
                                      T term, size_t n_rows, size_t n_cols) {
  size_t row = blockIdx.x;
  if (row >= n_rows) return;

  int cat = cats[row];
  size_t sim_row = idx_in[row];

  // Phase 1: compute exp(term * (1 - sim)) * penalty and accumulate sum
  T local_sum = T(0);
  for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
    T sim = similarities[sim_row * n_cols + col];
    T val;
    if constexpr (std::is_same<T, float>::value)
      val = expf(term * (T(1) - sim));
    else
      val = exp(term * (T(1) - sim));
    val *= penalty[(size_t)cat * n_cols + col];
    R_out[row * n_cols + col] = val;
    local_sum += val;
  }

// Phase 2: warp-level reduction
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

  // Phase 3: block-level reduction
  __shared__ T shared_sum[32];
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  if (lane_id == 0) shared_sum[warp_id] = local_sum;
  __syncthreads();

  T row_sum = T(0);
  if (threadIdx.x < 32) {
    int num_warps = (blockDim.x + 31) >> 5;
    if ((int)threadIdx.x < num_warps) row_sum = shared_sum[threadIdx.x];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
  }

  // Broadcast sum
  if (threadIdx.x == 0) shared_sum[0] = row_sum;
  __syncthreads();
  row_sum = shared_sum[0];

  // Phase 4: normalize
  T inv_sum = T(1) / row_sum;
  for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x)
    R_out[row * n_cols + col] *= inv_sum;
}
