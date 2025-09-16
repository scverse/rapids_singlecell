#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_scatter.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_scatter_add(std::uintptr_t v, std::uintptr_t cats, std::size_t n_cells,
                                      std::size_t n_pcs, std::size_t switcher, std::uintptr_t a) {
  dim3 block(256);
  std::size_t N = n_cells * n_pcs;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  scatter_add_kernel_optimized<T><<<grid, block>>>(reinterpret_cast<const T*>(v),
                                                   reinterpret_cast<const int*>(cats), n_cells,
                                                   n_pcs, switcher, reinterpret_cast<T*>(a));
}

template <typename T>
static inline void launch_aggregated_matrix(std::uintptr_t aggregated_matrix, std::uintptr_t sum,
                                            double top_corner, int n_batches) {
  dim3 block(32);
  dim3 grid((n_batches + 1 + 31) / 32);
  aggregated_matrix_kernel<T><<<grid, block>>>(reinterpret_cast<T*>(aggregated_matrix),
                                               reinterpret_cast<const T*>(sum), (T)top_corner,
                                               n_batches);
}

template <typename T>
static inline void launch_scatter_add_cat0(std::uintptr_t v, int n_cells, int n_pcs,
                                           std::uintptr_t a, std::uintptr_t bias) {
  dim3 block(1024);
  dim3 grid((n_pcs + 1) / 2, 8);
  scatter_add_kernel_with_bias_cat0<T><<<grid, block>>>(reinterpret_cast<const T*>(v), n_cells,
                                                        n_pcs, reinterpret_cast<T*>(a),
                                                        reinterpret_cast<const T*>(bias));
}

template <typename T>
static inline void launch_scatter_add_block(std::uintptr_t v, std::uintptr_t cat_offsets,
                                            std::uintptr_t cell_indices, int n_cells, int n_pcs,
                                            int n_batches, std::uintptr_t a, std::uintptr_t bias) {
  dim3 block(1024);
  dim3 grid(n_batches * ((n_pcs + 1) / 2));
  scatter_add_kernel_with_bias_block<T>
      <<<grid, block>>>(reinterpret_cast<const T*>(v), reinterpret_cast<const int*>(cat_offsets),
                        reinterpret_cast<const int*>(cell_indices), n_cells, n_pcs, n_batches,
                        reinterpret_cast<T*>(a), reinterpret_cast<const T*>(bias));
}

NB_MODULE(_harmony_scatter_cuda, m) {
  m.def("scatter_add", [](std::uintptr_t v, std::uintptr_t cats, std::size_t n_cells,
                          std::size_t n_pcs, std::size_t switcher, std::uintptr_t a, int itemsize) {
    if (itemsize == 4) {
      launch_scatter_add<float>(v, cats, n_cells, n_pcs, switcher, a);
    } else if (itemsize == 8) {
      launch_scatter_add<double>(v, cats, n_cells, n_pcs, switcher, a);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });

  m.def("aggregated_matrix", [](std::uintptr_t aggregated_matrix, std::uintptr_t sum,
                                double top_corner, int n_batches, int itemsize) {
    if (itemsize == 4) {
      launch_aggregated_matrix<float>(aggregated_matrix, sum, top_corner, n_batches);
    } else if (itemsize == 8) {
      launch_aggregated_matrix<double>(aggregated_matrix, sum, top_corner, n_batches);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });

  m.def("scatter_add_cat0", [](std::uintptr_t v, int n_cells, int n_pcs, std::uintptr_t a,
                               std::uintptr_t bias, int itemsize) {
    if (itemsize == 4) {
      launch_scatter_add_cat0<float>(v, n_cells, n_pcs, a, bias);
    } else if (itemsize == 8) {
      launch_scatter_add_cat0<double>(v, n_cells, n_pcs, a, bias);
    } else {
      throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
    }
  });

  m.def("scatter_add_block",
        [](std::uintptr_t v, std::uintptr_t cat_offsets, std::uintptr_t cell_indices, int n_cells,
           int n_pcs, int n_batches, std::uintptr_t a, std::uintptr_t bias, int itemsize) {
          if (itemsize == 4) {
            launch_scatter_add_block<float>(v, cat_offsets, cell_indices, n_cells, n_pcs, n_batches,
                                            a, bias);
          } else if (itemsize == 8) {
            launch_scatter_add_block<double>(v, cat_offsets, cell_indices, n_cells, n_pcs,
                                             n_batches, a, bias);
          } else {
            throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
          }
        });
}
