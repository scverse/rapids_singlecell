#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_scatter.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_scatter_add(const T* v, const int* cats, size_t n_cells, size_t n_pcs,
                                      size_t switcher, T* a, cudaStream_t stream) {
  dim3 block(256);
  size_t N = n_cells * n_pcs;
  dim3 grid((unsigned)((N + block.x - 1) / block.x));
  scatter_add_kernel<T><<<grid, block, 0, stream>>>(v, cats, n_cells, n_pcs, switcher, a);
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
static inline void launch_scatter_add_shared(const T* v, const int* cats, int n_cells, int n_pcs,
                                             int n_batches, int switcher, T* a, int n_blocks,
                                             cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(n_blocks);
  size_t shared_mem = (size_t)n_batches * n_pcs * sizeof(T);
  scatter_add_shared_kernel<T>
      <<<grid, block, shared_mem, stream>>>(v, cats, n_cells, n_pcs, n_batches, switcher, a);
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

template <typename T>
static inline void launch_gather_rows(const T* src, const int* idx, T* dst, int n_rows, int n_cols,
                                      cudaStream_t stream) {
  int n = n_rows * n_cols;
  gather_rows_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(src, idx, dst, n_rows, n_cols);
}

template <typename T>
static inline void launch_scatter_rows(const T* src, const int* idx, T* dst, int n_rows, int n_cols,
                                       cudaStream_t stream) {
  int n = n_rows * n_cols;
  scatter_rows_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(src, idx, dst, n_rows, n_cols);
}

static inline void launch_gather_int(const int* src, const int* idx, int* dst, int n,
                                     cudaStream_t stream) {
  gather_int_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, idx, dst, n);
}

NB_MODULE(_harmony_scatter_cuda, m) {
  // scatter_add - float32
  m.def(
      "scatter_add",
      [](cuda_array_c<const float> v, cuda_array_c<const int> cats, size_t n_cells, size_t n_pcs,
         size_t switcher, cuda_array_c<float> a, std::uintptr_t stream) {
        launch_scatter_add<float>(v.data(), cats.data(), n_cells, n_pcs, switcher, a.data(),
                                  (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "switcher"_a, "a"_a, "stream"_a = 0);

  // scatter_add - float64
  m.def(
      "scatter_add",
      [](cuda_array_c<const double> v, cuda_array_c<const int> cats, size_t n_cells, size_t n_pcs,
         size_t switcher, cuda_array_c<double> a, std::uintptr_t stream) {
        launch_scatter_add<double>(v.data(), cats.data(), n_cells, n_pcs, switcher, a.data(),
                                   (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "switcher"_a, "a"_a, "stream"_a = 0);

  // aggregated_matrix - float32
  m.def(
      "aggregated_matrix",
      [](cuda_array_c<float> aggregated_matrix, cuda_array_c<const float> sum, float top_corner,
         int n_batches, std::uintptr_t stream) {
        launch_aggregated_matrix<float>(aggregated_matrix.data(), sum.data(), top_corner, n_batches,
                                        (cudaStream_t)stream);
      },
      "aggregated_matrix"_a, nb::kw_only(), "sum"_a, "top_corner"_a, "n_batches"_a, "stream"_a = 0);

  // aggregated_matrix - float64
  m.def(
      "aggregated_matrix",
      [](cuda_array_c<double> aggregated_matrix, cuda_array_c<const double> sum, double top_corner,
         int n_batches, std::uintptr_t stream) {
        launch_aggregated_matrix<double>(aggregated_matrix.data(), sum.data(), top_corner,
                                         n_batches, (cudaStream_t)stream);
      },
      "aggregated_matrix"_a, nb::kw_only(), "sum"_a, "top_corner"_a, "n_batches"_a, "stream"_a = 0);

  // scatter_add_shared - float32
  m.def(
      "scatter_add_shared",
      [](cuda_array_c<const float> v, cuda_array_c<const int> cats, int n_cells, int n_pcs,
         int n_batches, int switcher, cuda_array_c<float> a, int n_blocks, std::uintptr_t stream) {
        launch_scatter_add_shared<float>(v.data(), cats.data(), n_cells, n_pcs, n_batches, switcher,
                                         a.data(), n_blocks, (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "n_batches"_a, "switcher"_a, "a"_a,
      "n_blocks"_a, "stream"_a = 0);

  // scatter_add_shared - float64
  m.def(
      "scatter_add_shared",
      [](cuda_array_c<const double> v, cuda_array_c<const int> cats, int n_cells, int n_pcs,
         int n_batches, int switcher, cuda_array_c<double> a, int n_blocks, std::uintptr_t stream) {
        launch_scatter_add_shared<double>(v.data(), cats.data(), n_cells, n_pcs, n_batches,
                                          switcher, a.data(), n_blocks, (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "n_batches"_a, "switcher"_a, "a"_a,
      "n_blocks"_a, "stream"_a = 0);

  // scatter_add_cat0 - float32
  m.def(
      "scatter_add_cat0",
      [](cuda_array_c<const float> v, int n_cells, int n_pcs, cuda_array_c<float> a,
         cuda_array_c<const float> bias, std::uintptr_t stream) {
        launch_scatter_add_cat0<float>(v.data(), n_cells, n_pcs, a.data(), bias.data(),
                                       (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "n_cells"_a, "n_pcs"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // scatter_add_cat0 - float64
  m.def(
      "scatter_add_cat0",
      [](cuda_array_c<const double> v, int n_cells, int n_pcs, cuda_array_c<double> a,
         cuda_array_c<const double> bias, std::uintptr_t stream) {
        launch_scatter_add_cat0<double>(v.data(), n_cells, n_pcs, a.data(), bias.data(),
                                        (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "n_cells"_a, "n_pcs"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // scatter_add_block - float32
  m.def(
      "scatter_add_block",
      [](cuda_array_c<const float> v, cuda_array_c<const int> cat_offsets,
         cuda_array_c<const int> cell_indices, int n_cells, int n_pcs, int n_batches,
         cuda_array_c<float> a, cuda_array_c<const float> bias, std::uintptr_t stream) {
        launch_scatter_add_block<float>(v.data(), cat_offsets.data(), cell_indices.data(), n_cells,
                                        n_pcs, n_batches, a.data(), bias.data(),
                                        (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cat_offsets"_a, "cell_indices"_a, "n_cells"_a, "n_pcs"_a,
      "n_batches"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // scatter_add_block - float64
  m.def(
      "scatter_add_block",
      [](cuda_array_c<const double> v, cuda_array_c<const int> cat_offsets,
         cuda_array_c<const int> cell_indices, int n_cells, int n_pcs, int n_batches,
         cuda_array_c<double> a, cuda_array_c<const double> bias, std::uintptr_t stream) {
        launch_scatter_add_block<double>(v.data(), cat_offsets.data(), cell_indices.data(), n_cells,
                                         n_pcs, n_batches, a.data(), bias.data(),
                                         (cudaStream_t)stream);
      },
      "v"_a, nb::kw_only(), "cat_offsets"_a, "cell_indices"_a, "n_cells"_a, "n_pcs"_a,
      "n_batches"_a, "a"_a, "bias"_a, "stream"_a = 0);

  // gather_rows - float32
  m.def(
      "gather_rows",
      [](cuda_array_c<const float> src, cuda_array_c<const int> idx, cuda_array_c<float> dst,
         int n_rows, int n_cols, std::uintptr_t stream) {
        launch_gather_rows<float>(src.data(), idx.data(), dst.data(), n_rows, n_cols,
                                  (cudaStream_t)stream);
      },
      "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // gather_rows - float64
  m.def(
      "gather_rows",
      [](cuda_array_c<const double> src, cuda_array_c<const int> idx, cuda_array_c<double> dst,
         int n_rows, int n_cols, std::uintptr_t stream) {
        launch_gather_rows<double>(src.data(), idx.data(), dst.data(), n_rows, n_cols,
                                   (cudaStream_t)stream);
      },
      "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // scatter_rows - float32
  m.def(
      "scatter_rows",
      [](cuda_array_c<const float> src, cuda_array_c<const int> idx, cuda_array_c<float> dst,
         int n_rows, int n_cols, std::uintptr_t stream) {
        launch_scatter_rows<float>(src.data(), idx.data(), dst.data(), n_rows, n_cols,
                                   (cudaStream_t)stream);
      },
      "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // scatter_rows - float64
  m.def(
      "scatter_rows",
      [](cuda_array_c<const double> src, cuda_array_c<const int> idx, cuda_array_c<double> dst,
         int n_rows, int n_cols, std::uintptr_t stream) {
        launch_scatter_rows<double>(src.data(), idx.data(), dst.data(), n_rows, n_cols,
                                    (cudaStream_t)stream);
      },
      "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);

  // gather_int
  m.def(
      "gather_int",
      [](cuda_array_c<const int> src, cuda_array_c<const int> idx, cuda_array_c<int> dst, int n,
         std::uintptr_t stream) {
        launch_gather_int(src.data(), idx.data(), dst.data(), n, (cudaStream_t)stream);
      },
      "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n"_a, "stream"_a = 0);
}
