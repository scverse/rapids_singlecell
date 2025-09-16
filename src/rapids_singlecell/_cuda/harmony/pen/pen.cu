#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_pen.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_pen(std::uintptr_t R, std::uintptr_t penalty, std::uintptr_t cats,
                              std::size_t n_rows, std::size_t n_cols) {
  dim3 block(256);
  std::size_t N = n_rows * n_cols;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  pen_kernel<T><<<grid, block>>>(reinterpret_cast<T*>(R), reinterpret_cast<const T*>(penalty),
                                 reinterpret_cast<const int*>(cats), n_rows, n_cols);
}

NB_MODULE(_harmony_pen_cuda, m) {
  m.def("pen", [](std::uintptr_t R, std::uintptr_t penalty, std::uintptr_t cats, std::size_t n_rows,
                  std::size_t n_cols, int itemsize) {
    if (itemsize == 4) {
      launch_pen<float>(R, penalty, cats, n_rows, n_cols);
    } else if (itemsize == 8) {
      launch_pen<double>(R, penalty, cats, n_rows, n_cols);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });
}
