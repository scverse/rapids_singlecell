#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_scatter.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_scatter_add(const T* v, const int* cats, std::size_t n_cells,
                                      std::size_t n_pcs, std::size_t switcher, T* a,
                                      cudaStream_t stream) {
  dim3 block(256);
  std::size_t N = n_cells * n_pcs;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  scatter_add_kernel_optimized<T><<<grid, block, 0, stream>>>(v, cats, n_cells, n_pcs, switcher, a);
}

template <typename T>
static inline void launch_aggregated_matrix(T* aggregated_matrix, const T* sum, T top_corner,
                                            int n_batches, cudaStream_t stream) {
  dim3 block(32);
  dim3 grid((n_batches + 1 + 31) / 32);
  aggregated_matrix_kernel<T>
      <<<grid, block, 0, stream>>>(aggregated_matrix, sum, top_corner, n_batches);
}

template <typename T>
static inline void launch_scatter_add_cat0(const T* v, int n_cells, int n_pcs, T* a, const T* bias,
                                           cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid((n_pcs + 1) / 2, 8);
  scatter_add_kernel_with_bias_cat0<T><<<grid, block, 0, stream>>>(v, n_cells, n_pcs, a, bias);
}

template <typename T>
static inline void launch_scatter_add_block(const T* v, const int* cat_offsets,
                                            const int* cell_indices, int n_cells, int n_pcs,
                                            int n_batches, T* a, const T* bias,
                                            cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(n_batches * ((n_pcs + 1) / 2));
  scatter_add_kernel_with_bias_block<T><<<grid, block, 0, stream>>>(
      v, cat_offsets, cell_indices, n_cells, n_pcs, n_batches, a, bias);
}

NB_MODULE(_harmony_scatter_cuda, m) {
  // scatter_add - float32
  m.def(
      "scatter_add",
      [](cuda_array<const float> v, cuda_array<const int> cats, std::size_t n_cells,
         std::size_t n_pcs, std::size_t switcher, cuda_array<float> a, std::uintptr_t stream) {
        launch_scatter_add<float>(v.data(), cats.data(), n_cells, n_pcs, switcher, a.data(),
                                  (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "switcher"_a, "a"_a, "stream"_a = 0);

  // scatter_add - float64
  m.def(
      "scatter_add",
      [](cuda_array<const double> v, cuda_array<const int> cats, std::size_t n_cells,
         std::size_t n_pcs, std::size_t switcher, cuda_array<double> a, std::uintptr_t stream) {
        launch_scatter_add<double>(v.data(), cats.data(), n_cells, n_pcs, switcher, a.data(),
                                   (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "switcher"_a, "a"_a, "stream"_a = 0);

  // aggregated_matrix - float32
  m.def(
      "aggregated_matrix",
      [](cuda_array<float> aggregated_matrix, cuda_array<const float> sum, float top_corner,
         int n_batches, std::uintptr_t stream) {
        launch_aggregated_matrix<float>(aggregated_matrix.data(), sum.data(), top_corner, n_batches,
                                        (cudaStream_t)stream);
      },
      "aggregated_matrix"_a, nb::kw_only(), "sum"_a, "top_corner"_a, "n_batches"_a, "stream"_a = 0);

  // aggregated_matrix - float64
  m.def(
      "aggregated_matrix",
      [](cuda_array<double> aggregated_matrix, cuda_array<const double> sum, double top_corner,
         int n_batches, std::uintptr_t stream) {
        launch_aggregated_matrix<double>(aggregated_matrix.data(), sum.data(), top_corner,
                                         n_batches, (cudaStream_t)stream);
      },
      "aggregated_matrix"_a, nb::kw_only(), "sum"_a, "top_corner"_a, "n_batches"_a, "stream"_a = 0);

  // scatter_add_cat0 - float32
  m.def(
      "scatter_add_cat0",
      [](cuda_array<const float> v, int n_cells, int n_pcs, cuda_array<float> a,
         cuda_array<const float> bias, std::uintptr_t stream) {
        launch_scatter_add_cat0<float>(v.data(), n_cells, n_pcs, a.data(), bias.data(),
                                       (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "n_cells"_a, "n_pcs"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // scatter_add_cat0 - float64
  m.def(
      "scatter_add_cat0",
      [](cuda_array<const double> v, int n_cells, int n_pcs, cuda_array<double> a,
         cuda_array<const double> bias, std::uintptr_t stream) {
        launch_scatter_add_cat0<double>(v.data(), n_cells, n_pcs, a.data(), bias.data(),
                                        (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "n_cells"_a, "n_pcs"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // scatter_add_block - float32
  m.def(
      "scatter_add_block",
      [](cuda_array<const float> v, cuda_array<const int> cat_offsets,
         cuda_array<const int> cell_indices, int n_cells, int n_pcs, int n_batches,
         cuda_array<float> a, cuda_array<const float> bias, std::uintptr_t stream) {
        launch_scatter_add_block<float>(v.data(), cat_offsets.data(), cell_indices.data(), n_cells,
                                        n_pcs, n_batches, a.data(), bias.data(),
                                        (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cat_offsets"_a, "cell_indices"_a, "n_cells"_a, "n_pcs"_a,
      "n_batches"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // scatter_add_block - float64
  m.def(
      "scatter_add_block",
      [](cuda_array<const double> v, cuda_array<const int> cat_offsets,
         cuda_array<const int> cell_indices, int n_cells, int n_pcs, int n_batches,
         cuda_array<double> a, cuda_array<const double> bias, std::uintptr_t stream) {
        launch_scatter_add_block<double>(v.data(), cat_offsets.data(), cell_indices.data(), n_cells,
                                         n_pcs, n_batches, a.data(), bias.data(),
                                         (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cat_offsets"_a, "cell_indices"_a, "n_cells"_a, "n_pcs"_a,
      "n_batches"_a, "a"_a, "bias"_a, "stream"_a = 0);
}
