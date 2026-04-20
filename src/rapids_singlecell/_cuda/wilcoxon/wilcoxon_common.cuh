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

#include "../nb_types.h"  // for CUDA_CHECK_LAST_ERROR

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int N_STREAMS = 4;
constexpr int SUB_BATCH_COLS = 64;
constexpr int BEGIN_BIT = 0;
constexpr int END_BIT = 32;
// Default thread-per-block for utility kernels (extract, gather, offsets,
// etc.).
constexpr int UTIL_BLOCK_SIZE = 256;
// Scratch slots for warp-level reduction (one slot per warp, 32 warps max).
constexpr int WARP_REDUCE_BUF = 32;
// Max group size for the fused smem-sort rank kernel (Tier 1 fast path).
// Beyond this, fall back to CUB segmented sort + binary-search rank kernel.
constexpr int TIER1_GROUP_THRESHOLD = 2500;

// ---------------------------------------------------------------------------
// RAII guard for cudaHostRegister.  Unregisters on scope exit even when an
// exception unwinds — prevents leaked host pinning on stream-sync failures.
// ---------------------------------------------------------------------------
struct HostRegisterGuard {
    void* ptr = nullptr;

    HostRegisterGuard() = default;
    HostRegisterGuard(void* p, size_t bytes, unsigned int flags = 0) : ptr(p) {
        if (ptr) cudaHostRegister(ptr, bytes, flags);
    }
    ~HostRegisterGuard() {
        if (ptr) cudaHostUnregister(ptr);
    }
    HostRegisterGuard(const HostRegisterGuard&) = delete;
    HostRegisterGuard& operator=(const HostRegisterGuard&) = delete;
    HostRegisterGuard(HostRegisterGuard&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    HostRegisterGuard& operator=(HostRegisterGuard&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaHostUnregister(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
};

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

/** Fill linear segment offsets [0, stride, 2*stride, ..., n_segments*stride]
 *  on-device.  One thread per output slot. */
__global__ void fill_linear_offsets_kernel(int* __restrict__ out,
                                           int n_segments, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_segments) out[i] = i * stride;
}

/** Fill linear segment offsets [0, stride, 2*stride, ...] on device.
 *  Runs on the supplied stream so it doesn't serialize multi-stream pipelines.
 */
static inline void upload_linear_offsets(int* d_offsets, int n_segments,
                                         int stride, cudaStream_t stream) {
    int count = n_segments + 1;
    int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
    fill_linear_offsets_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
        d_offsets, n_segments, stride);
    CUDA_CHECK_LAST_ERROR(fill_linear_offsets_kernel);
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
        int m = lo + ((hi - lo) >> 1);
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
