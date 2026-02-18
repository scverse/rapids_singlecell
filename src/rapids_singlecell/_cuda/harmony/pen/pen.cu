#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_pen.cuh"

using namespace nb::literals;

template <typename T, typename IdxT>
static inline void launch_fused_pen_norm(const T* similarities, const T* penalty, const int* cats,
                                         const IdxT* idx_in, T* R_out, T term, int n_rows,
                                         int n_cols, cudaStream_t stream) {
  unsigned block_dim = std::min(256u, std::max(32u, ((unsigned)n_cols + 31u) / 32u * 32u));
  fused_pen_norm_kernel<T, IdxT><<<n_rows, block_dim, 0, stream>>>(
      similarities, penalty, cats, idx_in, R_out, term, n_rows, n_cols);
}

template <typename T>
static inline void launch_penalty(const T* E, const T* O, const T* theta, T* penalty, int n_batches,
                                  int n_clusters, cudaStream_t stream) {
  int total = n_batches * n_clusters;
  penalty_kernel<T>
      <<<(total + 255) / 256, 256, 0, stream>>>(E, O, theta, penalty, n_batches, n_clusters);
}

template <typename T>
static void register_fused_pen_norm(nb::module_& m) {
  m.def(
      "fused_pen_norm",
      [](cuda_array_c<const T> similarities, cuda_array_c<const T> penalty,
         cuda_array_c<const int> cats, cuda_array_c<const size_t> idx_in, cuda_array_c<T> R_out,
         double term, int n_rows, int n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm<T, size_t>(similarities.data(), penalty.data(), cats.data(),
                                         idx_in.data(), R_out.data(), static_cast<T>(term), n_rows,
                                         n_cols, (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}

template <typename T>
static void register_penalty(nb::module_& m) {
  m.def(
      "penalty",
      [](cuda_array_c<const T> E, cuda_array_c<const T> O, cuda_array_c<const T> theta,
         cuda_array_c<T> penalty, int n_batches, int n_clusters, std::uintptr_t stream) {
        launch_penalty<T>(E.data(), O.data(), theta.data(), penalty.data(), n_batches, n_clusters,
                          (cudaStream_t)stream);
      },
      "E"_a, nb::kw_only(), "O"_a, "theta"_a, "penalty"_a, "n_batches"_a, "n_clusters"_a,
      "stream"_a = 0);
}

template <typename T>
static void register_fused_pen_norm_int(nb::module_& m) {
  m.def(
      "fused_pen_norm_int",
      [](cuda_array_c<const T> similarities, cuda_array_c<const T> penalty,
         cuda_array_c<const int> cats, cuda_array_c<const int> idx_in, cuda_array_c<T> R_out,
         double term, int n_rows, int n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm<T, int>(similarities.data(), penalty.data(), cats.data(),
                                      idx_in.data(), R_out.data(), static_cast<T>(term), n_rows,
                                      n_cols, (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}

NB_MODULE(_harmony_pen_cuda, m) {
  register_fused_pen_norm<float>(m);
  register_fused_pen_norm<double>(m);
  register_penalty<float>(m);
  register_penalty<double>(m);

  // -- Test-only bindings below --
  // fused_pen_norm_int uses int32 indices (used internally by the C++ clustering loop).
  // The binding exists solely for unit testing.
  register_fused_pen_norm_int<float>(m);
  register_fused_pen_norm_int<double>(m);
}
