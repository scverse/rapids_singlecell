#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_pen.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_fused_pen_norm(const T* similarities, const T* penalty, const int* cats,
                                         const size_t* idx_in, T* R_out, T term, size_t n_rows,
                                         size_t n_cols, cudaStream_t stream) {
  // Scale block dimension with columns, minimum 32, max 256
  unsigned block_dim = std::min(256u, std::max(32u, ((unsigned)n_cols + 31u) / 32u * 32u));
  dim3 block(block_dim);
  dim3 grid((unsigned)n_rows);
  fused_pen_norm_kernel<T><<<grid, block, 0, stream>>>(similarities, penalty, cats, idx_in, R_out,
                                                       term, n_rows, n_cols);
}

template <typename T>
static inline void launch_penalty(const T* E, const T* O, const T* theta, T* penalty, int n_batches,
                                  int n_clusters, cudaStream_t stream) {
  int total = n_batches * n_clusters;
  penalty_kernel<T>
      <<<(total + 255) / 256, 256, 0, stream>>>(E, O, theta, penalty, n_batches, n_clusters);
}

template <typename T>
static inline void launch_fused_pen_norm_int(const T* similarities, const T* penalty,
                                             const int* cats, const int* idx_in, T* R_out, T term,
                                             int n_rows, int n_cols, cudaStream_t stream) {
  unsigned block_dim = std::min(256u, std::max(32u, ((unsigned)n_cols + 31u) / 32u * 32u));
  fused_pen_norm_kernel_int<T><<<n_rows, block_dim, 0, stream>>>(
      similarities, penalty, cats, idx_in, R_out, term, n_rows, n_cols);
}

NB_MODULE(_harmony_pen_cuda, m) {
  // fused_pen_norm - float32
  m.def(
      "fused_pen_norm",
      [](cuda_array_c<const float> similarities, cuda_array_c<const float> penalty,
         cuda_array_c<const int> cats, cuda_array_c<const size_t> idx_in, cuda_array_c<float> R_out,
         float term, size_t n_rows, size_t n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm<float>(similarities.data(), penalty.data(), cats.data(),
                                     idx_in.data(), R_out.data(), term, n_rows, n_cols,
                                     (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // fused_pen_norm - float64
  m.def(
      "fused_pen_norm",
      [](cuda_array_c<const double> similarities, cuda_array_c<const double> penalty,
         cuda_array_c<const int> cats, cuda_array_c<const size_t> idx_in,
         cuda_array_c<double> R_out, double term, size_t n_rows, size_t n_cols,
         std::uintptr_t stream) {
        launch_fused_pen_norm<double>(similarities.data(), penalty.data(), cats.data(),
                                      idx_in.data(), R_out.data(), term, n_rows, n_cols,
                                      (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // penalty - float32
  m.def(
      "penalty",
      [](cuda_array_c<const float> E, cuda_array_c<const float> O, cuda_array_c<const float> theta,
         cuda_array_c<float> penalty, int n_batches, int n_clusters, std::uintptr_t stream) {
        launch_penalty<float>(E.data(), O.data(), theta.data(), penalty.data(), n_batches,
                              n_clusters, (cudaStream_t)stream);
      },
      "E"_a, nb::kw_only(), "O"_a, "theta"_a, "penalty"_a, "n_batches"_a, "n_clusters"_a,
      "stream"_a = 0);

  // penalty - float64
  m.def(
      "penalty",
      [](cuda_array_c<const double> E, cuda_array_c<const double> O,
         cuda_array_c<const double> theta, cuda_array_c<double> penalty, int n_batches,
         int n_clusters, std::uintptr_t stream) {
        launch_penalty<double>(E.data(), O.data(), theta.data(), penalty.data(), n_batches,
                               n_clusters, (cudaStream_t)stream);
      },
      "E"_a, nb::kw_only(), "O"_a, "theta"_a, "penalty"_a, "n_batches"_a, "n_clusters"_a,
      "stream"_a = 0);

  // fused_pen_norm_int - float32
  m.def(
      "fused_pen_norm_int",
      [](cuda_array_c<const float> similarities, cuda_array_c<const float> penalty,
         cuda_array_c<const int> cats, cuda_array_c<const int> idx_in, cuda_array_c<float> R_out,
         float term, int n_rows, int n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm_int<float>(similarities.data(), penalty.data(), cats.data(),
                                         idx_in.data(), R_out.data(), term, n_rows, n_cols,
                                         (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // fused_pen_norm_int - float64
  m.def(
      "fused_pen_norm_int",
      [](cuda_array_c<const double> similarities, cuda_array_c<const double> penalty,
         cuda_array_c<const int> cats, cuda_array_c<const int> idx_in, cuda_array_c<double> R_out,
         double term, int n_rows, int n_cols, std::uintptr_t stream) {
        launch_fused_pen_norm_int<double>(similarities.data(), penalty.data(), cats.data(),
                                          idx_in.data(), R_out.data(), term, n_rows, n_cols,
                                          (cudaStream_t)stream);
      },
      "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a, "R_out"_a, "term"_a,
      "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}
