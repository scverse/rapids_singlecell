#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_normalize.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_normalize(T* X, long long rows, long long cols, cudaStream_t stream) {
  unsigned block_dim = std::min(256u, std::max(32u, ((unsigned)cols + 31u) / 32u * 32u));
  dim3 block(block_dim);
  dim3 grid(rows);
  normalize_kernel<T><<<grid, block, 0, stream>>>(X, rows, cols);
}

NB_MODULE(_harmony_normalize_cuda, m) {
  // normalize - float32
  m.def(
      "normalize",
      [](cuda_array_c<float> X, long long rows, long long cols, std::uintptr_t stream) {
        launch_normalize<float>(X.data(), rows, cols, (cudaStream_t)stream);
      },
      "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);

  // normalize - float64
  m.def(
      "normalize",
      [](cuda_array_c<double> X, long long rows, long long cols, std::uintptr_t stream) {
        launch_normalize<double>(X.data(), rows, cols, (cudaStream_t)stream);
      },
      "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);
}
