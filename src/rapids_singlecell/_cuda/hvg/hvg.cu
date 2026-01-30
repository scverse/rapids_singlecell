#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
using cuda_array = nb::ndarray<T, nb::device::cuda, nb::c_contig>;

template <typename T>
__global__ void expected_zeros_kernel(const T* __restrict__ scaled_means,
                                      const T* __restrict__ total_counts, T* __restrict__ expected,
                                      int n_genes, int n_cells) {
  int gene = blockDim.x * blockIdx.x + threadIdx.x;
  if (gene >= n_genes) return;

  T sm = scaled_means[gene];
  T sum = T(0.0);

  for (int c = 0; c < n_cells; c++) {
    sum += exp(-sm * total_counts[c]);
  }

  expected[gene] = sum / T(n_cells);
}

template <typename T>
static void launch_expected_zeros(const T* scaled_means, const T* total_counts, T* expected,
                                  int n_genes, int n_cells, cudaStream_t stream) {
  int block_size = 256;
  int grid_size = (n_genes + block_size - 1) / block_size;
  expected_zeros_kernel<T><<<grid_size, block_size, 0, stream>>>(scaled_means, total_counts,
                                                                 expected, n_genes, n_cells);
}

NB_MODULE(_hvg_cuda, m) {
  m.def(
      "expected_zeros_f32",
      [](cuda_array<const float> scaled_means, cuda_array<const float> total_counts,
         cuda_array<float> expected, int n_genes, int n_cells, std::uintptr_t stream) {
        launch_expected_zeros<float>(scaled_means.data(), total_counts.data(), expected.data(),
                                     n_genes, n_cells, reinterpret_cast<cudaStream_t>(stream));
      },
      "scaled_means"_a, "total_counts"_a, "expected"_a, "n_genes"_a, "n_cells"_a, "stream"_a = 0);

  m.def(
      "expected_zeros_f64",
      [](cuda_array<const double> scaled_means, cuda_array<const double> total_counts,
         cuda_array<double> expected, int n_genes, int n_cells, std::uintptr_t stream) {
        launch_expected_zeros<double>(scaled_means.data(), total_counts.data(), expected.data(),
                                      n_genes, n_cells, reinterpret_cast<cudaStream_t>(stream));
      },
      "scaled_means"_a, "total_counts"_a, "expected"_a, "n_genes"_a, "n_cells"_a, "stream"_a = 0);
}
