#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_pv.cuh"

using namespace nb::literals;

static inline void launch_rev_cummin64(const double* x, double* y, int n_rows, int m,
                                       cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((unsigned)((n_rows + block.x - 1) / block.x));
  rev_cummin64_kernel<<<grid, block, 0, stream>>>(x, y, n_rows, m);
}

NB_MODULE(_pv_cuda, m) {
  m.def(
      "rev_cummin64",
      [](cuda_array_c<const double> x, cuda_array_c<double> out, int n_rows, int m,
         std::uintptr_t stream) {
        launch_rev_cummin64(x.data(), out.data(), n_rows, m, (cudaStream_t)stream);
      },
      "x"_a, nb::kw_only(), "out"_a, "n_rows"_a, "m"_a, "stream"_a = 0);
}
