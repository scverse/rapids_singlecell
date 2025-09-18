#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_pv.cuh"

namespace nb = nanobind;
using namespace nb::literals;

static inline void launch_rev_cummin64(std::uintptr_t x, std::uintptr_t y, int n_rows, int m,
                                       cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((unsigned)((n_rows + block.x - 1) / block.x));
  rev_cummin64_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const double*>(x),
                                                  reinterpret_cast<double*>(y), n_rows, m);
}

NB_MODULE(_pv_cuda, m) {
  m.def(
      "rev_cummin64",
      [](std::uintptr_t x, std::uintptr_t y, int n_rows, int m, std::uintptr_t stream) {
        launch_rev_cummin64(x, y, n_rows, m, (cudaStream_t)stream);
      },
      "x"_a, "y"_a, "n_rows"_a, "m"_a, "stream"_a = 0);
}
