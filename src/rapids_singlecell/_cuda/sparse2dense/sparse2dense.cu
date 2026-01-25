#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels_s2d.cuh"

namespace nb = nanobind;
using namespace nb::literals;

// Templated cuda_array over dtype and contiguity
template <typename T, typename Contig>
using cuda_array = nb::ndarray<T, nb::device::cuda, Contig>;

// Fully templated kernel launch - no runtime branches
template <typename T, bool C_ORDER>
static inline void launch_sparse2dense(const int* indptr, const int* index, const T* data, T* out,
                                       long long major, long long minor, int max_nnz,
                                       cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((unsigned)((major + block.x - 1) / block.x),
            (unsigned)((max_nnz + block.y - 1) / block.y));
  sparse2dense_kernel<T, C_ORDER>
      <<<grid, block, 0, stream>>>(indptr, index, data, out, major, minor);
}

// Runtime dispatch wrapper - c_switch depends on both sparse format and output order
template <typename T>
static inline void dispatch_sparse2dense(const int* indptr, const int* index, const T* data, T* out,
                                         long long major, long long minor, bool c_switch,
                                         int max_nnz, cudaStream_t stream) {
  if (c_switch) {
    launch_sparse2dense<T, true>(indptr, index, data, out, major, minor, max_nnz, stream);
  } else {
    launch_sparse2dense<T, false>(indptr, index, data, out, major, minor, max_nnz, stream);
  }
}

// Helper to define sparse2dense for a given dtype and output contiguity
template <typename T, typename OutContig>
void def_sparse2dense(nb::module_& m) {
  m.def(
      "sparse2dense",
      [](cuda_array<const int, nb::c_contig> indptr, cuda_array<const int, nb::c_contig> index,
         cuda_array<const T, nb::c_contig> data, cuda_array<T, OutContig> out, long long major,
         long long minor, bool c_switch, int max_nnz, std::uintptr_t stream) {
        dispatch_sparse2dense<T>(indptr.data(), index.data(), data.data(), out.data(), major, minor,
                                 c_switch, max_nnz, (cudaStream_t)stream);
      },
      "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out"_a, "major"_a, "minor"_a, "c_switch"_a,
      "max_nnz"_a, "stream"_a = 0);
}

NB_MODULE(_sparse2dense_cuda, m) {
  // F-order output must come before C-order for proper dispatch
  def_sparse2dense<float, nb::f_contig>(m);
  def_sparse2dense<float, nb::c_contig>(m);
  def_sparse2dense<double, nb::f_contig>(m);
  def_sparse2dense<double, nb::c_contig>(m);
}
