#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <cstdint>

#include "kernels_outer.cuh"

namespace nb = nanobind;

template <typename T>
static inline void launch_outer(std::uintptr_t E, std::uintptr_t Pr_b, std::uintptr_t R_sum,
                                long long n_cats, long long n_pcs, long long switcher,
                                cudaStream_t stream) {
  dim3 block(256);
  long long N = n_cats * n_pcs;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  outer_kernel<T>
      <<<grid, block, 0, stream>>>(reinterpret_cast<T*>(E), reinterpret_cast<const T*>(Pr_b),
                                   reinterpret_cast<const T*>(R_sum), n_cats, n_pcs, switcher);
}

template <typename T>
static inline void launch_harmony_corr(std::uintptr_t Z, std::uintptr_t W, std::uintptr_t cats,
                                       std::uintptr_t R, long long n_cells, long long n_pcs,
                                       cudaStream_t stream) {
  dim3 block(256);
  long long N = n_cells * n_pcs;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  harmony_correction_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<T*>(Z), reinterpret_cast<const T*>(W), reinterpret_cast<const int*>(cats),
      reinterpret_cast<const T*>(R), n_cells, n_pcs);
}

NB_MODULE(_harmony_outer_cuda, m) {
  m.def(
      "outer",
      [](std::uintptr_t E, std::uintptr_t Pr_b, std::uintptr_t R_sum, long long n_cats,
         long long n_pcs, long long switcher, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_outer<float>(E, Pr_b, R_sum, n_cats, n_pcs, switcher, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_outer<double>(E, Pr_b, R_sum, n_cats, n_pcs, switcher, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      nb::arg("E"), nb::arg("Pr_b"), nb::arg("R_sum"), nb::arg("n_cats"), nb::arg("n_pcs"),
      nb::arg("switcher"), nb::arg("itemsize"), nb::arg("stream") = 0);

  m.def(
      "harmony_corr",
      [](std::uintptr_t Z, std::uintptr_t W, std::uintptr_t cats, std::uintptr_t R,
         long long n_cells, long long n_pcs, int itemsize, std::uintptr_t stream) {
        if (itemsize == 4) {
          launch_harmony_corr<float>(Z, W, cats, R, n_cells, n_pcs, (cudaStream_t)stream);
        } else if (itemsize == 8) {
          launch_harmony_corr<double>(Z, W, cats, R, n_cells, n_pcs, (cudaStream_t)stream);
        } else {
          throw nb::value_error("Unsupported itemsize (expected 4 or 8)");
        }
      },
      nb::arg("Z"), nb::arg("W"), nb::arg("cats"), nb::arg("R"), nb::arg("n_cells"),
      nb::arg("n_pcs"), nb::arg("itemsize"), nb::arg("stream") = 0);
}
