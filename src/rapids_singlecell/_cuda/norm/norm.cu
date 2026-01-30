#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_norm.cuh"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
static inline void launch_dense_row_scale(T* data, int nrows, int ncols, T target_sum,
                                          cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((nrows + block.x - 1) / block.x);
  dense_row_scale_kernel<T><<<grid, block, 0, stream>>>(data, nrows, ncols, target_sum);
}

template <typename T>
static inline void launch_csr_row_scale(const int* indptr, T* data, int nrows, T target_sum,
                                        cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((nrows + block.x - 1) / block.x);
  csr_row_scale_kernel<T><<<grid, block, 0, stream>>>(indptr, data, nrows, target_sum);
}

template <typename T>
static inline void launch_csr_sum_major(const int* indptr, const T* data, T* sums, int major,
                                        cudaStream_t stream) {
  dim3 block(64);
  dim3 grid(major);
  std::size_t smem = static_cast<std::size_t>(block.x) * sizeof(T);
  csr_sum_major_kernel<T><<<grid, block, smem, stream>>>(indptr, data, sums, major);
}

NB_MODULE(_norm_cuda, m) {
  // mul_dense - float32
  m.def(
      "mul_dense",
      [](cuda_array<float> data, int nrows, int ncols, float target_sum, std::uintptr_t stream) {
        launch_dense_row_scale<float>(data.data(), nrows, ncols, target_sum, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "target_sum"_a, "stream"_a = 0);

  // mul_dense - float64
  m.def(
      "mul_dense",
      [](cuda_array<double> data, int nrows, int ncols, double target_sum, std::uintptr_t stream) {
        launch_dense_row_scale<double>(data.data(), nrows, ncols, target_sum, (cudaStream_t)stream);
      },
      "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "target_sum"_a, "stream"_a = 0);

  // mul_csr - float32
  m.def(
      "mul_csr",
      [](cuda_array<const int> indptr, cuda_array<float> data, int nrows, float target_sum,
         std::uintptr_t stream) {
        launch_csr_row_scale<float>(indptr.data(), data.data(), nrows, target_sum,
                                    (cudaStream_t)stream);
      },
      "indptr"_a, "data"_a, nb::kw_only(), "nrows"_a, "target_sum"_a, "stream"_a = 0);

  // mul_csr - float64
  m.def(
      "mul_csr",
      [](cuda_array<const int> indptr, cuda_array<double> data, int nrows, double target_sum,
         std::uintptr_t stream) {
        launch_csr_row_scale<double>(indptr.data(), data.data(), nrows, target_sum,
                                     (cudaStream_t)stream);
      },
      "indptr"_a, "data"_a, nb::kw_only(), "nrows"_a, "target_sum"_a, "stream"_a = 0);

  // sum_major - float32
  m.def(
      "sum_major",
      [](cuda_array<const int> indptr, cuda_array<const float> data, cuda_array<float> sums,
         int major, std::uintptr_t stream) {
        launch_csr_sum_major<float>(indptr.data(), data.data(), sums.data(), major,
                                    (cudaStream_t)stream);
      },
      "indptr"_a, "data"_a, nb::kw_only(), "sums"_a, "major"_a, "stream"_a = 0);

  // sum_major - float64
  m.def(
      "sum_major",
      [](cuda_array<const int> indptr, cuda_array<const double> data, cuda_array<double> sums,
         int major, std::uintptr_t stream) {
        launch_csr_sum_major<double>(indptr.data(), data.data(), sums.data(), major,
                                     (cudaStream_t)stream);
      },
      "indptr"_a, "data"_a, nb::kw_only(), "sums"_a, "major"_a, "stream"_a = 0);
}
