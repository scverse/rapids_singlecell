#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_nmf.cuh"

using namespace nb::literals;

constexpr int HALS_BLOCK_MAX = 256;
constexpr int HALS_BLOCK_MIN = 32;

/// Pick block size for the cached kernel variant.
/// Returns 0 if shared memory is insufficient (→ use local-memory fallback).
static inline int pick_cached_block(int K, int elem_size) {
    int device;
    cudaGetDevice(&device);
    int max_shared;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock,
                           device);

    size_t gram_bytes = (size_t)K * K * elem_size;
    if ((int)gram_bytes >= max_shared) return 0;
    size_t avail = max_shared - gram_bytes;
    int max_threads = (int)(avail / (K * elem_size));
    int block = (max_threads / 32) * 32;
    if (block > HALS_BLOCK_MAX) block = HALS_BLOCK_MAX;
    return (block >= HALS_BLOCK_MIN) ? block : 0;
}

template <typename T>
static inline void launch_hals_update_H(T* H, const T* WtX, const T* WtW, int m,
                                        int K, T l1_reg, T l2_reg, int n_sweeps,
                                        cudaStream_t stream) {
    int block = pick_cached_block(K, sizeof(T));

    if (block > 0) {
        dim3 grid((m + block - 1) / block);
        size_t smem = (size_t)K * K * sizeof(T) + block * K * sizeof(T);
        hals_update_H_cached_kernel<T><<<grid, block, smem, stream>>>(
            H, WtX, WtW, m, K, l1_reg, l2_reg, n_sweeps);
        CUDA_CHECK_LAST_ERROR(hals_update_H_cached_kernel);
    } else {
        int fb_block = HALS_BLOCK_MAX;
        dim3 grid((m + fb_block - 1) / fb_block);
        size_t smem = (size_t)K * K * sizeof(T);
        hals_update_H_local_kernel<T><<<grid, fb_block, smem, stream>>>(
            H, WtX, WtW, m, K, l1_reg, l2_reg, n_sweeps);
        CUDA_CHECK_LAST_ERROR(hals_update_H_local_kernel);
    }
}

template <typename T>
static inline void launch_hals_update_W(T* W, const T* XHt, const T* HHt, int n,
                                        int K, T l1_reg, T l2_reg, int n_sweeps,
                                        cudaStream_t stream) {
    int block = pick_cached_block(K, sizeof(T));

    if (block > 0) {
        dim3 grid((n + block - 1) / block);
        size_t smem = (size_t)K * K * sizeof(T) + block * K * sizeof(T);
        hals_update_W_cached_kernel<T><<<grid, block, smem, stream>>>(
            W, XHt, HHt, n, K, l1_reg, l2_reg, n_sweeps);
        CUDA_CHECK_LAST_ERROR(hals_update_W_cached_kernel);
    } else {
        int fb_block = HALS_BLOCK_MAX;
        dim3 grid((n + fb_block - 1) / fb_block);
        size_t smem = (size_t)K * K * sizeof(T);
        hals_update_W_local_kernel<T><<<grid, fb_block, smem, stream>>>(
            W, XHt, HHt, n, K, l1_reg, l2_reg, n_sweeps);
        CUDA_CHECK_LAST_ERROR(hals_update_W_local_kernel);
    }
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "hals_update_H",
        [](gpu_array_c<float, Device> H, gpu_array_c<const float, Device> WtX,
           gpu_array_c<const float, Device> WtW, int m_dim, int K, float l1_reg,
           float l2_reg, int n_sweeps, std::uintptr_t stream) {
            launch_hals_update_H<float>(H.data(), WtX.data(), WtW.data(), m_dim,
                                        K, l1_reg, l2_reg, n_sweeps,
                                        (cudaStream_t)stream);
        },
        "H"_a, "WtX"_a, "WtW"_a, nb::kw_only(), "m"_a, "K"_a, "l1_reg"_a,
        "l2_reg"_a, "n_sweeps"_a = 1, "stream"_a = 0);

    m.def(
        "hals_update_H",
        [](gpu_array_c<double, Device> H, gpu_array_c<const double, Device> WtX,
           gpu_array_c<const double, Device> WtW, int m_dim, int K,
           double l1_reg, double l2_reg, int n_sweeps, std::uintptr_t stream) {
            launch_hals_update_H<double>(H.data(), WtX.data(), WtW.data(),
                                         m_dim, K, l1_reg, l2_reg, n_sweeps,
                                         (cudaStream_t)stream);
        },
        "H"_a, "WtX"_a, "WtW"_a, nb::kw_only(), "m"_a, "K"_a, "l1_reg"_a,
        "l2_reg"_a, "n_sweeps"_a = 1, "stream"_a = 0);

    m.def(
        "hals_update_W",
        [](gpu_array_c<float, Device> W, gpu_array_c<const float, Device> XHt,
           gpu_array_c<const float, Device> HHt, int n, int K, float l1_reg,
           float l2_reg, int n_sweeps, std::uintptr_t stream) {
            launch_hals_update_W<float>(W.data(), XHt.data(), HHt.data(), n, K,
                                        l1_reg, l2_reg, n_sweeps,
                                        (cudaStream_t)stream);
        },
        "W"_a, "XHt"_a, "HHt"_a, nb::kw_only(), "n"_a, "K"_a, "l1_reg"_a,
        "l2_reg"_a, "n_sweeps"_a = 1, "stream"_a = 0);

    m.def(
        "hals_update_W",
        [](gpu_array_c<double, Device> W, gpu_array_c<const double, Device> XHt,
           gpu_array_c<const double, Device> HHt, int n, int K, double l1_reg,
           double l2_reg, int n_sweeps, std::uintptr_t stream) {
            launch_hals_update_W<double>(W.data(), XHt.data(), HHt.data(), n, K,
                                         l1_reg, l2_reg, n_sweeps,
                                         (cudaStream_t)stream);
        },
        "W"_a, "XHt"_a, "HHt"_a, nb::kw_only(), "n"_a, "K"_a, "l1_reg"_a,
        "l2_reg"_a, "n_sweeps"_a = 1, "stream"_a = 0);
}

NB_MODULE(_nmf_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
