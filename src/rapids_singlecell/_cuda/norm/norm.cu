#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_norm.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_dense_row_scale(T* data, int nrows, int ncols,
                                          T target_sum, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(nrows);
    dense_row_scale_kernel<T>
        <<<grid, block, 0, stream>>>(data, nrows, ncols, target_sum);
}

template <typename T>
static inline void launch_csr_row_scale(const int* indptr, T* data, int nrows,
                                        T target_sum, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(nrows);
    csr_row_scale_kernel<T>
        <<<grid, block, 0, stream>>>(indptr, data, nrows, target_sum);
}

template <typename T>
static inline void launch_csr_sum_major(const int* indptr, const T* data,
                                        T* sums, int major,
                                        cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(major);
    csr_sum_major_kernel<T>
        <<<grid, block, 0, stream>>>(indptr, data, sums, major);
}

template <typename T>
static inline void launch_find_hi_genes_csr(const int* indptr,
                                            const int* indices, const T* data,
                                            bool* gene_is_hi, T max_fraction,
                                            int nrows, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(nrows);
    find_hi_genes_csr_kernel<T><<<grid, block, 0, stream>>>(
        indptr, indices, data, gene_is_hi, max_fraction, nrows);
}

template <typename T>
static inline void launch_masked_mul_csr(const int* indptr, const int* indices,
                                         T* data, const bool* gene_mask,
                                         int nrows, T tsum,
                                         cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(nrows);
    masked_mul_csr_kernel<T><<<grid, block, 0, stream>>>(
        indptr, indices, data, gene_mask, nrows, tsum);
}

template <typename T>
static inline void launch_masked_sum_major(const int* indptr,
                                           const int* indices, const T* data,
                                           const bool* gene_mask, T* sums,
                                           int major, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(major);
    masked_sum_major_kernel<T><<<grid, block, 0, stream>>>(
        indptr, indices, data, gene_mask, sums, major);
}

template <typename T>
static inline void launch_prescaled_mul_csr(const int* indptr, T* data,
                                            const T* scales, int nrows,
                                            cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(nrows);
    prescaled_mul_csr_kernel<T>
        <<<grid, block, 0, stream>>>(indptr, data, scales, nrows);
}

template <typename T>
static inline void launch_prescaled_mul_dense(T* data, const T* scales,
                                              int nrows, int ncols,
                                              cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(nrows);
    prescaled_mul_dense_kernel<T>
        <<<grid, block, 0, stream>>>(data, scales, nrows, ncols);
}

NB_MODULE(_norm_cuda, m) {
    // mul_dense - float32
    m.def(
        "mul_dense",
        [](cuda_array_c<float> data, int nrows, int ncols, float target_sum,
           std::uintptr_t stream) {
            launch_dense_row_scale<float>(data.data(), nrows, ncols, target_sum,
                                          (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "target_sum"_a,
        "stream"_a = 0);

    // mul_dense - float64
    m.def(
        "mul_dense",
        [](cuda_array_c<double> data, int nrows, int ncols, double target_sum,
           std::uintptr_t stream) {
            launch_dense_row_scale<double>(data.data(), nrows, ncols,
                                           target_sum, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "target_sum"_a,
        "stream"_a = 0);

    // mul_csr - float32
    m.def(
        "mul_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<float> data, int nrows,
           float target_sum, std::uintptr_t stream) {
            launch_csr_row_scale<float>(indptr.data(), data.data(), nrows,
                                        target_sum, (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "nrows"_a, "target_sum"_a,
        "stream"_a = 0);

    // mul_csr - float64
    m.def(
        "mul_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<double> data, int nrows,
           double target_sum, std::uintptr_t stream) {
            launch_csr_row_scale<double>(indptr.data(), data.data(), nrows,
                                         target_sum, (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "nrows"_a, "target_sum"_a,
        "stream"_a = 0);

    // sum_major - float32
    m.def(
        "sum_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const float> data,
           cuda_array_c<float> sums, int major, std::uintptr_t stream) {
            launch_csr_sum_major<float>(indptr.data(), data.data(), sums.data(),
                                        major, (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "sums"_a, "major"_a,
        "stream"_a = 0);

    // sum_major - float64
    m.def(
        "sum_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const double> data,
           cuda_array_c<double> sums, int major, std::uintptr_t stream) {
            launch_csr_sum_major<double>(indptr.data(), data.data(),
                                         sums.data(), major,
                                         (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "sums"_a, "major"_a,
        "stream"_a = 0);

    // find_hi_genes_csr - float32
    m.def(
        "find_hi_genes_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<const float> data, cuda_array_c<bool> gene_is_hi,
           float max_fraction, int nrows, std::uintptr_t stream) {
            launch_find_hi_genes_csr<float>(
                indptr.data(), indices.data(), data.data(), gene_is_hi.data(),
                max_fraction, nrows, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_is_hi"_a,
        "max_fraction"_a, "nrows"_a, "stream"_a = 0);

    // find_hi_genes_csr - float64
    m.def(
        "find_hi_genes_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<const double> data, cuda_array_c<bool> gene_is_hi,
           double max_fraction, int nrows, std::uintptr_t stream) {
            launch_find_hi_genes_csr<double>(
                indptr.data(), indices.data(), data.data(), gene_is_hi.data(),
                max_fraction, nrows, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_is_hi"_a,
        "max_fraction"_a, "nrows"_a, "stream"_a = 0);

    // masked_mul_csr - float32
    m.def(
        "masked_mul_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<float> data, cuda_array_c<const bool> gene_mask,
           int nrows, float tsum, std::uintptr_t stream) {
            launch_masked_mul_csr<float>(indptr.data(), indices.data(),
                                         data.data(), gene_mask.data(), nrows,
                                         tsum, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_mask"_a,
        "nrows"_a, "tsum"_a, "stream"_a = 0);

    // masked_mul_csr - float64
    m.def(
        "masked_mul_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<double> data, cuda_array_c<const bool> gene_mask,
           int nrows, double tsum, std::uintptr_t stream) {
            launch_masked_mul_csr<double>(indptr.data(), indices.data(),
                                          data.data(), gene_mask.data(), nrows,
                                          tsum, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_mask"_a,
        "nrows"_a, "tsum"_a, "stream"_a = 0);

    // masked_sum_major - float32
    m.def(
        "masked_sum_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<const float> data, cuda_array_c<const bool> gene_mask,
           cuda_array_c<float> sums, int major, std::uintptr_t stream) {
            launch_masked_sum_major<float>(
                indptr.data(), indices.data(), data.data(), gene_mask.data(),
                sums.data(), major, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_mask"_a,
        "sums"_a, "major"_a, "stream"_a = 0);

    // masked_sum_major - float64
    m.def(
        "masked_sum_major",
        [](cuda_array_c<const int> indptr, cuda_array_c<const int> indices,
           cuda_array_c<const double> data, cuda_array_c<const bool> gene_mask,
           cuda_array_c<double> sums, int major, std::uintptr_t stream) {
            launch_masked_sum_major<double>(
                indptr.data(), indices.data(), data.data(), gene_mask.data(),
                sums.data(), major, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_mask"_a,
        "sums"_a, "major"_a, "stream"_a = 0);

    // prescaled_mul_csr - float32
    m.def(
        "prescaled_mul_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<float> data,
           cuda_array_c<const float> scales, int nrows, std::uintptr_t stream) {
            launch_prescaled_mul_csr<float>(indptr.data(), data.data(),
                                            scales.data(), nrows,
                                            (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "scales"_a, "nrows"_a,
        "stream"_a = 0);

    // prescaled_mul_csr - float64
    m.def(
        "prescaled_mul_csr",
        [](cuda_array_c<const int> indptr, cuda_array_c<double> data,
           cuda_array_c<const double> scales, int nrows,
           std::uintptr_t stream) {
            launch_prescaled_mul_csr<double>(indptr.data(), data.data(),
                                             scales.data(), nrows,
                                             (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "scales"_a, "nrows"_a,
        "stream"_a = 0);

    // prescaled_mul_dense - float32
    m.def(
        "prescaled_mul_dense",
        [](cuda_array_c<float> data, cuda_array_c<const float> scales,
           int nrows, int ncols, std::uintptr_t stream) {
            launch_prescaled_mul_dense<float>(data.data(), scales.data(), nrows,
                                              ncols, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "scales"_a, "nrows"_a, "ncols"_a,
        "stream"_a = 0);

    // prescaled_mul_dense - float64
    m.def(
        "prescaled_mul_dense",
        [](cuda_array_c<double> data, cuda_array_c<const double> scales,
           int nrows, int ncols, std::uintptr_t stream) {
            launch_prescaled_mul_dense<double>(
                data.data(), scales.data(), nrows, ncols, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "scales"_a, "nrows"_a, "ncols"_a,
        "stream"_a = 0);
}
