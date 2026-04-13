#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <rmm/device_buffer.hpp>
#if __has_include(<rmm/mr/per_device_resource.hpp>)
#include <rmm/mr/per_device_resource.hpp>  // rmm >= 26.02
#else
#include <rmm/mr/device/per_device_resource.hpp>  // rmm 25.x
#endif

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int N_STREAMS = 4;
constexpr int SUB_BATCH_COLS = 32;
constexpr int BEGIN_BIT = 0;
constexpr int END_BIT = 32;

// ---------------------------------------------------------------------------
// RMM pool helper — allocate GPU buffers through the current RMM memory
// resource.  Buffers are stored in a vector and freed (RAII) when the vector
// is destroyed.
// ---------------------------------------------------------------------------
struct RmmPool {
    std::vector<rmm::device_buffer> bufs;
    rmm::device_async_resource_ref mr;

    RmmPool() : mr(rmm::mr::get_current_device_resource()) {
    }

    template <typename T>
    T* alloc(size_t count) {
        bufs.emplace_back(count * sizeof(T), rmm::cuda_stream_default, mr);
        return static_cast<T*>(bufs.back().data());
    }
};

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

/** Upload linear segment offsets [0, stride, 2*stride, ...] to device.
 *  Uses synchronous copy — the buffer is small (a few hundred bytes). */
static inline void upload_linear_offsets(int* d_offsets, int n_segments,
                                         int stride, cudaStream_t stream) {
    std::vector<int> h(n_segments + 1);
    for (int i = 0; i <= n_segments; i++) h[i] = i * stride;
    cudaMemcpy(d_offsets, h.data(), (n_segments + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
}

// ============================================================================
// CSR → dense F-order extraction (templated on data type)
// ============================================================================

template <typename T>
__global__ void csr_extract_dense_kernel(const T* __restrict__ data,
                                         const int* __restrict__ indices,
                                         const int* __restrict__ indptr,
                                         const int* __restrict__ row_ids,
                                         T* __restrict__ out, int n_target,
                                         int col_start, int col_stop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_target) return;

    int row = row_ids[tid];
    int rs = indptr[row];
    int re = indptr[row + 1];

    int lo = rs, hi = re;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (indices[m] < col_start)
            lo = m + 1;
        else
            hi = m;
    }

    for (int p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        out[(long long)(c - col_start) * n_target + tid] = data[p];
    }
}
