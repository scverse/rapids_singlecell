#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_cooc.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

static inline void launch_count_pairwise(const float* spatial, const float* thresholds,
                                         const int* labels, int* result, int n, int k, int l_val,
                                         cudaStream_t stream) {
  dim3 grid(n);
  dim3 block(32);
  occur_count_kernel_pairwise<<<grid, block, 0, stream>>>(spatial, thresholds, labels, result, n, k,
                                                          l_val);
}

static inline bool launch_reduce_shared(const int* result, float* out, int k, int l_val, int format,
                                        cudaStream_t stream) {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (prop.sharedMemPerBlock <
      static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1) * sizeof(float)) {
    return false;
  }

  dim3 grid(l_val);
  dim3 block(32);
  std::size_t smem = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1) * sizeof(float);
  occur_reduction_kernel_shared<<<grid, block, smem, stream>>>(result, out, k, l_val, format);
  return true;
}

static inline void launch_reduce_global(const int* result, float* inter_out, float* out, int k,
                                        int l_val, int format, cudaStream_t stream) {
  dim3 grid(l_val);
  dim3 block(32);
  std::size_t smem = static_cast<std::size_t>(k) * sizeof(float);
  occur_reduction_kernel_global<<<grid, block, smem, stream>>>(result, inter_out, out, k, l_val,
                                                               format);
}

static inline bool launch_count_csr_catpairs_auto(const float* spatial, const float* thresholds,
                                                  const int* cat_offsets, const int* cell_indices,
                                                  const int* pair_left, const int* pair_right,
                                                  int* counts_delta, int num_pairs, int k,
                                                  int l_val, cudaStream_t stream) {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int l_pad = ((l_val + 31) / 32) * 32;
  int chosen = 0;
  for (int tpb : {1024, 512, 256, 128, 64, 32}) {
    int warps = tpb / 32;
    std::size_t req =
        static_cast<std::size_t>(warps) * static_cast<std::size_t>(l_pad) * sizeof(int);
    if (req <= prop.sharedMemPerBlock) {
      chosen = tpb;
      break;
    }
  }
  if (chosen == 0) return false;

  std::size_t smem =
      static_cast<std::size_t>(chosen / 32) * static_cast<std::size_t>(l_pad) * sizeof(int);
  dim3 grid(num_pairs);
  dim3 block(chosen);
  occur_count_kernel_csr_catpairs<<<grid, block, smem, stream>>>(
      spatial, thresholds, cat_offsets, cell_indices, pair_left, pair_right, counts_delta, k,
      l_val);
  return true;
}

NB_MODULE(_cooc_cuda, m) {
  m.def(
      "count_pairwise",
      [](cuda_array<const float> spatial, cuda_array<const float> thresholds,
         cuda_array<const int> labels, cuda_array<int> result, int n, int k, int l_val,
         std::uintptr_t stream) {
        launch_count_pairwise(spatial.data(), thresholds.data(), labels.data(), result.data(), n, k,
                              l_val, (cudaStream_t)stream);
      },
      "spatial"_a, nb::kw_only(), "thresholds"_a, "labels"_a, "result"_a, "n"_a, "k"_a, "l_val"_a,
      "stream"_a = 0);

  m.def(
      "reduce_shared",
      [](cuda_array<const int> result, cuda_array<float> out, int k, int l_val, int format,
         std::uintptr_t stream) {
        return launch_reduce_shared(result.data(), out.data(), k, l_val, format,
                                    (cudaStream_t)stream);
      },
      "result"_a, nb::kw_only(), "out"_a, "k"_a, "l_val"_a, "format"_a, "stream"_a = 0);

  m.def(
      "reduce_global",
      [](cuda_array<const int> result, cuda_array<float> inter_out, cuda_array<float> out, int k,
         int l_val, int format, std::uintptr_t stream) {
        launch_reduce_global(result.data(), inter_out.data(), out.data(), k, l_val, format,
                             (cudaStream_t)stream);
      },
      "result"_a, nb::kw_only(), "inter_out"_a, "out"_a, "k"_a, "l_val"_a, "format"_a,
      "stream"_a = 0);

  m.def(
      "count_csr_catpairs_auto",
      [](cuda_array<const float> spatial, cuda_array<const float> thresholds,
         cuda_array<const int> cat_offsets, cuda_array<const int> cell_indices,
         cuda_array<const int> pair_left, cuda_array<const int> pair_right,
         cuda_array<int> counts_delta, int num_pairs, int k, int l_val, std::uintptr_t stream) {
        return launch_count_csr_catpairs_auto(spatial.data(), thresholds.data(), cat_offsets.data(),
                                              cell_indices.data(), pair_left.data(),
                                              pair_right.data(), counts_delta.data(), num_pairs, k,
                                              l_val, (cudaStream_t)stream);
      },
      "spatial"_a, nb::kw_only(), "thresholds"_a, "cat_offsets"_a, "cell_indices"_a, "pair_left"_a,
      "pair_right"_a, "counts_delta"_a, "num_pairs"_a, "k"_a, "l_val"_a, "stream"_a = 0);
}
