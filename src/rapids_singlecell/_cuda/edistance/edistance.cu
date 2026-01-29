#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include "kernels_edistance.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

// Tile sizes for feature dimension
static constexpr int TILE_SIZES[] = {32, 50, 64};
static constexpr int NUM_TILE_SIZES = 3;

// Choose optimal feat_tile based on n_features and shared memory limits
static int choose_feat_tile(int n_features, size_t max_shared_bytes, int cell_tile,
                            int dtype_size) {
  // Shared memory: cell_tile * feat_tile * dtype_size + warp_sums overhead
  size_t warp_sums_overhead = 32 * dtype_size;
  size_t available_shared = max_shared_bytes - warp_sums_overhead;

  int best_tile = 32;  // default minimum

  // Check exact divisibility - prefer larger tiles
  for (int i = NUM_TILE_SIZES - 1; i >= 0; --i) {
    int tile = TILE_SIZES[i];
    size_t required = static_cast<size_t>(cell_tile) * tile * dtype_size;
    if (required <= available_shared) {
      if (n_features % tile == 0) {
        return tile;
      }
      if (best_tile == 32 || tile > best_tile) {
        best_tile = tile;
      }
    }
  }

  return best_tile;
}

// Get kernel configuration for given parameters
// Returns (cell_tile, feat_tile, block_size, shared_mem_bytes) or None if insufficient memory
static nb::object get_kernel_config(int n_features, bool is_double) {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  int dtype_size = is_double ? 8 : 4;
  int cell_tile = is_double ? 16 : 32;

  int feat_tile = choose_feat_tile(n_features, prop.sharedMemPerBlock, cell_tile, dtype_size);

  // Default block size for Ampere+ (CC >= 8)
  int block_size = 1024;

  // For pre-Ampere GPUs with float64, reduce block size
  if (is_double && prop.major < 8) {
    block_size = 256;
  }

  // Shared memory: smem_b (cell_tile * feat_tile)
  size_t shared_mem_bytes = static_cast<size_t>(cell_tile) * feat_tile * dtype_size;

  if (shared_mem_bytes > prop.sharedMemPerBlock) {
    return nb::none();
  }

  return nb::make_tuple(cell_tile, feat_tile, block_size, static_cast<int>(shared_mem_bytes));
}

// Launch kernel with specific tile sizes
template <typename T, int CELL_TILE, int FEAT_TILE>
static void launch_edistance_kernel(const T* embedding, const int* cat_offsets,
                                    const int* cell_indices, const int* pair_left,
                                    const int* pair_right, T* pairwise_sums, int num_pairs, int k,
                                    int n_features, int blocks_per_pair, int block_size,
                                    size_t shared_mem, cudaStream_t stream) {
  dim3 grid(num_pairs, blocks_per_pair);
  dim3 block(block_size);
  edistance_kernel<T, CELL_TILE, FEAT_TILE><<<grid, block, shared_mem, stream>>>(
      embedding, cat_offsets, cell_indices, pair_left, pair_right, pairwise_sums, k, n_features,
      blocks_per_pair);
}

// Dispatch to correct tile size specialization for float32
static void dispatch_f32(const float* embedding, const int* cat_offsets, const int* cell_indices,
                         const int* pair_left, const int* pair_right, float* pairwise_sums,
                         int num_pairs, int k, int n_features, int blocks_per_pair, int feat_tile,
                         int block_size, size_t shared_mem, cudaStream_t stream) {
  // cell_tile is 32 for float32
  if (feat_tile == 64) {
    launch_edistance_kernel<float, 32, 64>(embedding, cat_offsets, cell_indices, pair_left,
                                           pair_right, pairwise_sums, num_pairs, k, n_features,
                                           blocks_per_pair, block_size, shared_mem, stream);
  } else if (feat_tile == 50) {
    launch_edistance_kernel<float, 32, 50>(embedding, cat_offsets, cell_indices, pair_left,
                                           pair_right, pairwise_sums, num_pairs, k, n_features,
                                           blocks_per_pair, block_size, shared_mem, stream);
  } else {
    launch_edistance_kernel<float, 32, 32>(embedding, cat_offsets, cell_indices, pair_left,
                                           pair_right, pairwise_sums, num_pairs, k, n_features,
                                           blocks_per_pair, block_size, shared_mem, stream);
  }
}

// Dispatch to correct tile size specialization for float64
static void dispatch_f64(const double* embedding, const int* cat_offsets, const int* cell_indices,
                         const int* pair_left, const int* pair_right, double* pairwise_sums,
                         int num_pairs, int k, int n_features, int blocks_per_pair, int feat_tile,
                         int block_size, size_t shared_mem, cudaStream_t stream) {
  // cell_tile is 16 for float64
  if (feat_tile == 64) {
    launch_edistance_kernel<double, 16, 64>(embedding, cat_offsets, cell_indices, pair_left,
                                            pair_right, pairwise_sums, num_pairs, k, n_features,
                                            blocks_per_pair, block_size, shared_mem, stream);
  } else if (feat_tile == 50) {
    launch_edistance_kernel<double, 16, 50>(embedding, cat_offsets, cell_indices, pair_left,
                                            pair_right, pairwise_sums, num_pairs, k, n_features,
                                            blocks_per_pair, block_size, shared_mem, stream);
  } else {
    launch_edistance_kernel<double, 16, 32>(embedding, cat_offsets, cell_indices, pair_left,
                                            pair_right, pairwise_sums, num_pairs, k, n_features,
                                            blocks_per_pair, block_size, shared_mem, stream);
  }
}

NB_MODULE(_edistance_cuda, m) {
  m.def("get_kernel_config", &get_kernel_config, "n_features"_a, "is_double"_a,
        "Get kernel configuration (cell_tile, feat_tile, block_size, shared_mem) for given "
        "parameters. Returns None if insufficient shared memory.");

  m.def(
      "compute_distances_f32",
      [](cuda_array<const float> embedding, cuda_array<const int> cat_offsets,
         cuda_array<const int> cell_indices, cuda_array<const int> pair_left,
         cuda_array<const int> pair_right, cuda_array<float> pairwise_sums, int num_pairs, int k,
         int n_features, int blocks_per_pair, int feat_tile, int block_size, int shared_mem,
         std::uintptr_t stream) {
        dispatch_f32(embedding.data(), cat_offsets.data(), cell_indices.data(), pair_left.data(),
                     pair_right.data(), pairwise_sums.data(), num_pairs, k, n_features,
                     blocks_per_pair, feat_tile, block_size, static_cast<size_t>(shared_mem),
                     reinterpret_cast<cudaStream_t>(stream));
      },
      "embedding"_a, "cat_offsets"_a, "cell_indices"_a, "pair_left"_a, "pair_right"_a,
      "pairwise_sums"_a, "num_pairs"_a, "k"_a, "n_features"_a, "blocks_per_pair"_a, "feat_tile"_a,
      "block_size"_a, "shared_mem"_a, "stream"_a = 0);

  m.def(
      "compute_distances_f64",
      [](cuda_array<const double> embedding, cuda_array<const int> cat_offsets,
         cuda_array<const int> cell_indices, cuda_array<const int> pair_left,
         cuda_array<const int> pair_right, cuda_array<double> pairwise_sums, int num_pairs, int k,
         int n_features, int blocks_per_pair, int feat_tile, int block_size, int shared_mem,
         std::uintptr_t stream) {
        dispatch_f64(embedding.data(), cat_offsets.data(), cell_indices.data(), pair_left.data(),
                     pair_right.data(), pairwise_sums.data(), num_pairs, k, n_features,
                     blocks_per_pair, feat_tile, block_size, static_cast<size_t>(shared_mem),
                     reinterpret_cast<cudaStream_t>(stream));
      },
      "embedding"_a, "cat_offsets"_a, "cell_indices"_a, "pair_left"_a, "pair_right"_a,
      "pairwise_sums"_a, "num_pairs"_a, "k"_a, "n_features"_a, "blocks_per_pair"_a, "feat_tile"_a,
      "block_size"_a, "shared_mem"_a, "stream"_a = 0);
}
