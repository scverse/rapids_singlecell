#pragma once

#include <cuda_runtime.h>

/// Machine epsilon for float/double, usable in device code.
template <typename T>
struct EpsTrait;

template <>
struct EpsTrait<float> {
    static constexpr float value = 1.1920929e-7f;
};

template <>
struct EpsTrait<double> {
    static constexpr double value = 2.2204460492503131e-16;
};

/// Maximum K for the local-memory fallback kernels.
constexpr int HALS_MAX_K = 200;

// -----------------------------------------------------------------------
// H kernels  (K x m, each thread owns one column)
// -----------------------------------------------------------------------

/// Cached variant — per-thread H column in shared memory.
///
/// Shared layout: [K*K WtW gram] [blockDim*K column cache]
template <typename T>
__global__ void hals_update_H_cached_kernel(T* __restrict__ H,
                                            const T* __restrict__ WtX,
                                            const T* __restrict__ WtW,
                                            const int m, const int K,
                                            const T l1_reg, const T l2_reg,
                                            const int n_sweeps) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr T EPS = EpsTrait<T>::value;

    extern __shared__ char smem_raw[];
    T* sWtW = reinterpret_cast<T*>(smem_raw);
    T* h_cache = reinterpret_cast<T*>(smem_raw) + K * K + threadIdx.x * K;

    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x)
        sWtW[idx] = WtW[idx];
    __syncthreads();

    if (col >= m) return;

    for (int j = 0; j < K; j++) h_cache[j] = H[j * m + col];

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        for (int kk = 0; kk < K; kk++) {
            T dot = T(0);
            for (int j = 0; j < K; j++) dot += sWtW[kk * K + j] * h_cache[j];
            T diag = sWtW[kk * K + kk] + l2_reg;
            if (diag < EPS) diag = EPS;
            T val = h_cache[kk] +
                    (WtX[kk * m + col] - dot - l1_reg - l2_reg * h_cache[kk]) /
                        diag;
            h_cache[kk] = val > T(0) ? val : T(0);
        }
    }

    for (int j = 0; j < K; j++) H[j * m + col] = h_cache[j];
}

/// Fallback — gram in shared, H column in thread-local memory (L1-backed).
/// Used when K is too large for the per-thread shared cache.
/// Always runs at full blockDim (256).
template <typename T>
__global__ void hals_update_H_local_kernel(T* __restrict__ H,
                                           const T* __restrict__ WtX,
                                           const T* __restrict__ WtW,
                                           const int m, const int K,
                                           const T l1_reg, const T l2_reg,
                                           const int n_sweeps) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr T EPS = EpsTrait<T>::value;

    extern __shared__ char smem_raw[];
    T* sWtW = reinterpret_cast<T*>(smem_raw);

    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x)
        sWtW[idx] = WtW[idx];
    __syncthreads();

    if (col >= m) return;

    T h_local[HALS_MAX_K];
    for (int j = 0; j < K; j++) h_local[j] = H[j * m + col];

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        for (int kk = 0; kk < K; kk++) {
            T dot = T(0);
            for (int j = 0; j < K; j++) dot += sWtW[kk * K + j] * h_local[j];
            T diag = sWtW[kk * K + kk] + l2_reg;
            if (diag < EPS) diag = EPS;
            T val = h_local[kk] +
                    (WtX[kk * m + col] - dot - l1_reg - l2_reg * h_local[kk]) /
                        diag;
            h_local[kk] = val > T(0) ? val : T(0);
        }
    }

    for (int j = 0; j < K; j++) H[j * m + col] = h_local[j];
}

// -----------------------------------------------------------------------
// W kernels  (n x K, each thread owns one row)
// -----------------------------------------------------------------------

/// Cached variant — per-thread W row in shared memory.
///
/// Shared layout: [K*K HHt gram] [blockDim*K row cache]
template <typename T>
__global__ void hals_update_W_cached_kernel(T* __restrict__ W,
                                            const T* __restrict__ XHt,
                                            const T* __restrict__ HHt,
                                            const int n, const int K,
                                            const T l1_reg, const T l2_reg,
                                            const int n_sweeps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr T EPS = EpsTrait<T>::value;

    extern __shared__ char smem_raw[];
    T* sHHt = reinterpret_cast<T*>(smem_raw);
    T* w_cache = reinterpret_cast<T*>(smem_raw) + K * K + threadIdx.x * K;

    // Transpose HHt into F-order so the j-loop reads are contiguous.
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x) {
        int r = idx / K;
        int c = idx % K;
        sHHt[c * K + r] = HHt[idx];
    }
    __syncthreads();

    if (row >= n) return;

    for (int j = 0; j < K; j++) w_cache[j] = W[row * K + j];

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        for (int kk = 0; kk < K; kk++) {
            T dot = T(0);
            for (int j = 0; j < K; j++) dot += w_cache[j] * sHHt[kk * K + j];
            T diag = sHHt[kk * K + kk] + l2_reg;
            if (diag < EPS) diag = EPS;
            T val = w_cache[kk] +
                    (XHt[row * K + kk] - dot - l1_reg - l2_reg * w_cache[kk]) /
                        diag;
            w_cache[kk] = val > T(0) ? val : T(0);
        }
    }

    for (int j = 0; j < K; j++) W[row * K + j] = w_cache[j];
}

/// Fallback — gram in shared, W row in thread-local memory (L1-backed).
/// Used when K is too large for the per-thread shared cache.
/// Always runs at full blockDim (256).
template <typename T>
__global__ void hals_update_W_local_kernel(T* __restrict__ W,
                                           const T* __restrict__ XHt,
                                           const T* __restrict__ HHt,
                                           const int n, const int K,
                                           const T l1_reg, const T l2_reg,
                                           const int n_sweeps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr T EPS = EpsTrait<T>::value;

    extern __shared__ char smem_raw[];
    T* sHHt = reinterpret_cast<T*>(smem_raw);

    // Transpose HHt into F-order so the j-loop reads are contiguous.
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x) {
        int r = idx / K;
        int c = idx % K;
        sHHt[c * K + r] = HHt[idx];
    }
    __syncthreads();

    if (row >= n) return;

    T w_local[HALS_MAX_K];
    for (int j = 0; j < K; j++) w_local[j] = W[row * K + j];

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        for (int kk = 0; kk < K; kk++) {
            T dot = T(0);
            for (int j = 0; j < K; j++) dot += w_local[j] * sHHt[kk * K + j];
            T diag = sHHt[kk * K + kk] + l2_reg;
            if (diag < EPS) diag = EPS;
            T val = w_local[kk] +
                    (XHt[row * K + kk] - dot - l1_reg - l2_reg * w_local[kk]) /
                        diag;
            w_local[kk] = val > T(0) ? val : T(0);
        }
    }

    for (int j = 0; j < K; j++) W[row * K + j] = w_local[j];
}
