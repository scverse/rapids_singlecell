#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_normalize.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_normalize(std::uintptr_t X, long long rows, long long cols) {
  dim3 block(32);
  dim3 grid(rows);
  normalize_kernel_optimized<T><<<grid, block>>>(reinterpret_cast<T*>(X), rows, cols);
}

NB_MODULE(_harmony_normalize_cuda, m) {
  m.def("normalize", [](std::uintptr_t X, long long rows, long long cols, int itemsize) {
    if (itemsize == 4) {
      launch_normalize<float>(X, rows, cols);
    } else if (itemsize == 8) {
      launch_normalize<double>(X, rows, cols);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });
}
