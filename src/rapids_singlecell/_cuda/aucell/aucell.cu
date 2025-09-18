#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

__global__ void auc_kernel(const int* __restrict__ ranks, int R, int C,
                           const int* __restrict__ cnct, const int* __restrict__ starts,
                           const int* __restrict__ lens, int n_sets, int n_up,
                           const float* __restrict__ max_aucs, float* __restrict__ es) {
  const int set = blockIdx.x;
  const int row = blockIdx.y * blockDim.x + threadIdx.x;
  if (set >= n_sets || row >= R) return;

  const int start = starts[set];
  const int end = start + lens[set];

  int r = 0;
  int s = 0;

  for (int i = start; i < end; ++i) {
    const int g = cnct[i];
    const int rk = ranks[row * C + g];
    if (rk <= n_up) {
      r += 1;
      s += rk;
    }
  }
  const float val = (float)((static_cast<long long>(r) * n_up) - s) / max_aucs[set];
  es[row * n_sets + set] = val;
}

static inline void launch_auc(std::uintptr_t ranks, int R, int C, std::uintptr_t cnct,
                              std::uintptr_t starts, std::uintptr_t lens, int n_sets, int n_up,
                              std::uintptr_t max_aucs, std::uintptr_t es, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((unsigned)n_sets, (unsigned)((R + block.x - 1) / block.x));
  auc_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const int*>(ranks), R, C, reinterpret_cast<const int*>(cnct),
      reinterpret_cast<const int*>(starts), reinterpret_cast<const int*>(lens), n_sets, n_up,
      reinterpret_cast<const float*>(max_aucs), reinterpret_cast<float*>(es));
}

NB_MODULE(_aucell_cuda, m) {
  m.def(
      "auc",
      [](std::uintptr_t ranks, int R, int C, std::uintptr_t cnct, std::uintptr_t starts,
         std::uintptr_t lens, int n_sets, int n_up, std::uintptr_t max_aucs, std::uintptr_t es,
         std::uintptr_t stream) {
        launch_auc(ranks, R, C, cnct, starts, lens, n_sets, n_up, max_aucs, es,
                   (cudaStream_t)stream);
      },
      "ranks"_a, "R"_a, "C"_a, "cnct"_a, "starts"_a, "lens"_a, "n_sets"_a, "n_up"_a, "max_aucs"_a,
      "es"_a, "stream"_a = 0);
}
