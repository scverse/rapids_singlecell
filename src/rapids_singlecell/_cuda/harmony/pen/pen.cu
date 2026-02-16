#include <cstddef>
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_pen.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_fused_pen_norm(const T* similarities, const T* penalty, const int* cats,
                                         const std::size_t* idx_in, T* R_out, T term,
                                         std::size_t n_rows, std::size_t n_cols,
                                         cudaStream_t stream) {
  // Scale block dimension with columns, minimum 32, max 256
  unsigned block_dim = std::min(256u, std::max(32u, ((unsigned)n_cols + 31u) / 32u * 32u));
  dim3 block(block_dim);
  dim3 grid((unsigned)n_rows);
  fused_pen_norm_kernel<T><<<grid, block, 0, stream>>>(similarities, penalty, cats, idx_in, R_out,
                                                       term, n_rows, n_cols);
}

NB_MODULE(_harmony_pen_cuda, m) {
  // fused_pen_norm - float32
  m.def(
      "fused_pen_norm",
      [](cuda_array<const float> similarities, cuda_array<const float> penalty,
         cuda_array<const int> cats, cuda_array<const std::size_t> idx_in, cuda_array<float> R_out,
         float term, std::size_t n_rows, std::size_t n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm<float>(similarities.data(), penalty.data(), cats.data(),
                                     idx_in.data(), R_out.data(), term, n_rows, n_cols,
                                     (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // fused_pen_norm - float64
  m.def(
      "fused_pen_norm",
      [](cuda_array<const double> similarities, cuda_array<const double> penalty,
         cuda_array<const int> cats, cuda_array<const std::size_t> idx_in, cuda_array<double> R_out,
         double term, std::size_t n_rows, std::size_t n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm<double>(similarities.data(), penalty.data(), cats.data(),
                                      idx_in.data(), R_out.data(), term, n_rows, n_cols,
                                      (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}
