#pragma once

#include <cuda_runtime.h>

// Templated kernel for computing pairwise group distances
// Supports both float and double precision
// Uses shared memory tiling over cells and features

template <typename T, int CELL_TILE, int FEAT_TILE>
__global__ void edistance_kernel(const T* __restrict__ embedding,
                                 const int* __restrict__ cat_offsets,
                                 const int* __restrict__ cell_indices,
                                 const int* __restrict__ pair_left,
                                 const int* __restrict__ pair_right, T* __restrict__ pairwise_sums,
                                 int k, int n_features, int blocks_per_pair) {
  // Shared memory for B tile: [FEAT_TILE][CELL_TILE]
  extern __shared__ char smem_raw[];
  T* smem_b = reinterpret_cast<T*>(smem_raw);

  const int thread_id = threadIdx.x;
  const int pair_id = blockIdx.x;
  const int block_in_pair = blockIdx.y;
  const int block_size = blockDim.x;

  T local_sum = T(0.0);

  const int a = pair_left[pair_id];
  const int b = pair_right[pair_id];

  const int start_a = cat_offsets[a];
  const int end_a = cat_offsets[a + 1];
  const int start_b = cat_offsets[b];
  const int end_b = cat_offsets[b + 1];

  const int n_a = end_a - start_a;
  const int n_b = end_b - start_b;

  // Distribute A cells across blocks_per_pair
  const int total_threads_for_pair = blocks_per_pair * block_size;
  const int global_thread_in_pair = block_in_pair * block_size + thread_id;
  const int n_iters_a = (n_a + total_threads_for_pair - 1) / total_threads_for_pair;

  for (int iter_a = 0; iter_a < n_iters_a; ++iter_a) {
    const int ia = start_a + iter_a * total_threads_for_pair + global_thread_in_pair;
    const bool valid_a = (ia < end_a);
    const int idx_i = valid_a ? cell_indices[ia] : 0;
    const int i_local = ia - start_a;

    // Tile over B cells
    for (int jb_base = 0; jb_base < n_b; jb_base += CELL_TILE) {
      const int cells_in_tile = min(CELL_TILE, n_b - jb_base);

      // Accumulate squared distances for this cell tile
      T dist_sq[CELL_TILE];
#pragma unroll
      for (int c = 0; c < CELL_TILE; ++c) dist_sq[c] = T(0.0);

      // Tile over features
      for (int feat_base = 0; feat_base < n_features; feat_base += FEAT_TILE) {
        const int feats_in_tile = min(FEAT_TILE, n_features - feat_base);

        // Cooperatively load B tile into shared memory
        const int total_elems = FEAT_TILE * CELL_TILE;
        for (int i = thread_id; i < total_elems; i += block_size) {
          int cell_idx = i / FEAT_TILE;
          int feat_idx = i % FEAT_TILE;
          T val = T(0.0);
          if (cell_idx < cells_in_tile && feat_idx < feats_in_tile) {
            int global_b_idx = cell_indices[start_b + jb_base + cell_idx];
            val = embedding[global_b_idx * n_features + feat_base + feat_idx];
          }
          // Store as smem_b[feat][cell] for sequential access
          smem_b[feat_idx * CELL_TILE + cell_idx] = val;
        }

        __syncthreads();

        // Compute partial squared differences for this feature chunk
        if (valid_a) {
          for (int f = 0; f < feats_in_tile; ++f) {
            T val_a = embedding[idx_i * n_features + feat_base + f];

#pragma unroll
            for (int c = 0; c < CELL_TILE; ++c) {
              T val_b = smem_b[f * CELL_TILE + c];
              T diff = val_a - val_b;
              dist_sq[c] += diff * diff;
            }
          }
        }

        __syncthreads();
      }

      // dist_sq[c] contains full squared distance for cell c
      // Accumulate sqrt(dist_sq) into local_sum
      if (valid_a) {
#pragma unroll
        for (int c = 0; c < CELL_TILE; ++c) {
          if (c >= cells_in_tile) break;
          int j_local = jb_base + c;

          // Skip lower triangle for diagonal blocks
          if (a == b && i_local >= j_local) continue;

          local_sum += sqrt(dist_sq[c]);
        }
      }
    }
  }

  // Warp shuffle reduction
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

  // Block reduction via shared memory
  static __shared__ T warp_sums[32];
  if ((thread_id & 31) == 0) warp_sums[thread_id >> 5] = local_sum;
  __syncthreads();

  if (thread_id < 32) {
    T val = (thread_id < (block_size >> 5)) ? warp_sums[thread_id] : T(0.0);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);

    if (thread_id == 0) {
      atomicAdd(&pairwise_sums[a * k + b], val);
      if (a != b) {
        atomicAdd(&pairwise_sums[b * k + a], val);
      }
    }
  }
}
