#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_cooc.cuh"

namespace nb = nanobind;

static inline void launch_count_pairwise(std::uintptr_t spatial, std::uintptr_t thresholds,
                                         std::uintptr_t labels, std::uintptr_t result, int n, int k,
                                         int l_val, cudaStream_t stream) {
  dim3 grid(n);
  dim3 block(32);
  occur_count_kernel_pairwise<<<grid, block, 0, stream>>>(
      reinterpret_cast<const float*>(spatial), reinterpret_cast<const float*>(thresholds),
      reinterpret_cast<const int*>(labels), reinterpret_cast<int*>(result), n, k, l_val);
}

static inline bool launch_reduce_shared(std::uintptr_t result, std::uintptr_t out, int k, int l_val,
                                        int format, cudaStream_t stream) {
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
  occur_reduction_kernel_shared<<<grid, block, smem, stream>>>(
      reinterpret_cast<const int*>(result), reinterpret_cast<float*>(out), k, l_val, format);
  return true;
}

static inline void launch_reduce_global(std::uintptr_t result, std::uintptr_t inter_out,
                                        std::uintptr_t out, int k, int l_val, int format,
                                        cudaStream_t stream) {
  dim3 grid(l_val);
  dim3 block(32);
  std::size_t smem = static_cast<std::size_t>(k) * sizeof(float);
  occur_reduction_kernel_global<<<grid, block, smem, stream>>>(
      reinterpret_cast<const int*>(result), reinterpret_cast<float*>(inter_out),
      reinterpret_cast<float*>(out), k, l_val, format);
}

// Auto-pick threads-per-block; return false if insufficient shared memory
static inline bool launch_count_csr_catpairs_auto(
    std::uintptr_t spatial, std::uintptr_t thresholds, std::uintptr_t cat_offsets,
    std::uintptr_t cell_indices, std::uintptr_t pair_left, std::uintptr_t pair_right,
    std::uintptr_t counts_delta, int num_pairs, int k, int l_val, cudaStream_t stream) {
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
  if (chosen == 0) {
    return false;
  }
  std::size_t smem =
      static_cast<std::size_t>(chosen / 32) * static_cast<std::size_t>(l_pad) * sizeof(int);
  dim3 grid(num_pairs);
  dim3 block(chosen);
  occur_count_kernel_csr_catpairs<<<grid, block, smem, stream>>>(
      reinterpret_cast<const float*>(spatial), reinterpret_cast<const float*>(thresholds),
      reinterpret_cast<const int*>(cat_offsets), reinterpret_cast<const int*>(cell_indices),
      reinterpret_cast<const int*>(pair_left), reinterpret_cast<const int*>(pair_right),
      reinterpret_cast<int*>(counts_delta), k, l_val);
  return true;
}

NB_MODULE(_cooc_cuda, m) {
  m.def(
      "count_pairwise",
      [](std::uintptr_t spatial, std::uintptr_t thresholds, std::uintptr_t labels,
         std::uintptr_t result, int n, int k, int l_val, std::uintptr_t stream) {
        launch_count_pairwise(spatial, thresholds, labels, result, n, k, l_val,
                              (cudaStream_t)stream);
      },
      nb::arg("spatial"), nb::arg("thresholds"), nb::arg("labels"), nb::arg("result"), nb::arg("n"),
      nb::arg("k"), nb::arg("l_val"), nb::arg("stream") = 0);
  m.def(
      "reduce_shared",
      [](std::uintptr_t result, std::uintptr_t out, int k, int l_val, int format,
         std::uintptr_t stream) {
        return launch_reduce_shared(result, out, k, l_val, format, (cudaStream_t)stream);
      },
      nb::arg("result"), nb::arg("out"), nb::arg("k"), nb::arg("l_val"), nb::arg("format"),
      nb::arg("stream") = 0);
  m.def(
      "reduce_global",
      [](std::uintptr_t result, std::uintptr_t inter_out, std::uintptr_t out, int k, int l_val,
         int format, std::uintptr_t stream) {
        launch_reduce_global(result, inter_out, out, k, l_val, format, (cudaStream_t)stream);
      },
      nb::arg("result"), nb::arg("inter_out"), nb::arg("out"), nb::arg("k"), nb::arg("l_val"),
      nb::arg("format"), nb::arg("stream") = 0);
  m.def(
      "count_csr_catpairs_auto",
      [](std::uintptr_t spatial, std::uintptr_t thresholds, std::uintptr_t cat_offsets,
         std::uintptr_t cell_indices, std::uintptr_t pair_left, std::uintptr_t pair_right,
         std::uintptr_t counts_delta, int num_pairs, int k, int l_val, std::uintptr_t stream) {
        return launch_count_csr_catpairs_auto(spatial, thresholds, cat_offsets, cell_indices,
                                              pair_left, pair_right, counts_delta, num_pairs, k,
                                              l_val, (cudaStream_t)stream);
      },
      nb::arg("spatial"), nb::arg("thresholds"), nb::arg("cat_offsets"), nb::arg("cell_indices"),
      nb::arg("pair_left"), nb::arg("pair_right"), nb::arg("counts_delta"), nb::arg("num_pairs"),
      nb::arg("k"), nb::arg("l_val"), nb::arg("stream") = 0);
}
