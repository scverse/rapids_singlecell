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
// Max group size for the super-fast "warp-per-(col,group)" fused kernel
// (Tier 0).  Each warp sorts and ranks one (col, group) pair entirely in
// registers via warp-shuffle bitonic sort — no smem sort buffer, no
// __syncthreads().  Blocks pack 8 warps so block launch overhead is
// amortised 8× across (col, group) work items.  This path is the fast
// route for per-celltype perturbation-style workloads where most test
// groups have only a few dozen cells.
constexpr int TIER0_GROUP_THRESHOLD = 32;
// Medium-group cutoff for the unsorted direct-rank kernel.  For perturbation
// workloads most groups sit below this range, where avoiding a full smem
// bitonic sort wins despite the O(n^2) in-group count.
constexpr int TIER2_GROUP_THRESHOLD = 512;
// Max group size for the fused smem-sort rank kernel (Tier 1 fast path).
// Beyond this, fall back to CUB segmented sort + binary-search rank kernel.
constexpr int TIER1_GROUP_THRESHOLD = 2500;
// Per-stream dense slab budget (float32 items).  Dynamic sub-batching sizes
// each group's column batch so that (n_g × eff_sb_cols) ≤ this.  Bigger =
// fewer kernel launches; smaller = less per-stream memory.  64M items × 4B =
// 256 MB per stream dense slab + same for sorted copy ≈ 512 MB / stream.
constexpr size_t GROUP_DENSE_BUDGET_ITEMS = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// RAII guard for cudaHostRegister.  Unregisters on scope exit even when an
// exception unwinds — prevents leaked host pinning on stream-sync failures.
// ---------------------------------------------------------------------------
struct HostRegisterGuard {
    void* ptr = nullptr;

    HostRegisterGuard() = default;
    HostRegisterGuard(void* p, size_t bytes, unsigned int flags = 0) {
        if (p && bytes > 0) {
            cudaError_t err = cudaHostRegister(p, bytes, flags);
            if (err != cudaSuccess) {
                // Already-registered memory is fine; anything else means the
                // subsequent kernels would read garbage from an unmapped
                // pointer, so surface the error immediately.
                if (err == cudaErrorHostMemoryAlreadyRegistered) {
                    cudaGetLastError();  // clear sticky error flag
                } else {
                    throw std::runtime_error(
                        std::string("cudaHostRegister failed (") +
                        std::to_string((size_t)bytes) +
                        " bytes, flags=" + std::to_string(flags) +
                        "): " + cudaGetErrorString(err));
                }
            } else {
                ptr = p;
            }
        }
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

/** Fill per-row stats codes for a pack of K groups.
 *  Given pack_grp_offsets (size K+1, relative to pack start), write
 *  stats_codes[r] = base_slot + group_idx_of_row_r for r in [0, pack_n_rows).
 *  Binary search within the K+1 offsets. */
__global__ void fill_pack_stats_codes_kernel(
    const int* __restrict__ pack_grp_offsets, int* __restrict__ stats_codes,
    int K, int base_slot) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int pack_n_rows = pack_grp_offsets[K];
    if (r >= pack_n_rows) return;
    int lo = 0, hi = K;
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (pack_grp_offsets[m + 1] <= r)
            lo = m + 1;
        else
            hi = m;
    }
    stats_codes[r] = base_slot + lo;
}

/** Rebase a slice of indptr: out[i] = indptr[col + i] - indptr[col].
 *  Grid-strided: supports arbitrary `count` (no single-block thread limit).
 *  Templated so that 64-bit global indptrs can produce 32-bit pack-local
 *  indptrs (per-pack nnz always fits in int32 thanks to the memory budget).
 */
template <typename IdxIn, typename IdxOut>
__global__ void rebase_indptr_kernel(const IdxIn* __restrict__ indptr,
                                     IdxOut* __restrict__ out, int col,
                                     int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (IdxOut)(indptr[col + i] - indptr[col]);
}

/** Fused gather + cast-to-float32 + stats accumulation, reading from mapped
 *  pinned host memory.  Block-per-row; threads in the block cooperate on the
 *  row's nnz.  Each nnz is read from host over PCIe exactly once — no
 *  intermediate native-dtype GPU buffer, no second GPU pass.
 *
 *  h_data / h_indices: device-accessible pointers into mapped pinned host
 *                     memory (cudaHostRegisterMapped).
 *  d_indptr_full: full-matrix indptr on device.
 *  d_row_ids:    rows to gather (size n_target_rows).
 *  d_out_indptr: pre-computed compacted indptr, size n_target_rows+1 with
 *                out_indptr[i+1] - out_indptr[i] equal to the source row's
 *                nnz.
 *
 *  Slot dispatch:
 *    d_stats_codes != nullptr → slot = d_stats_codes[r]; otherwise slot =
 *    fixed_slot (used for the Ref phase where every row maps to the same
 *    slot).  slot ∉ [0, n_groups_stats) skips accumulation.
 */
template <typename InT, typename IndexT, typename IndptrT>
__global__ void csr_gather_cast_accumulate_mapped_kernel(
    const InT* __restrict__ h_data, const IndexT* __restrict__ h_indices,
    const IndptrT* __restrict__ d_indptr_full,
    const int* __restrict__ d_row_ids, const int* __restrict__ d_out_indptr,
    const int* __restrict__ d_stats_codes, int fixed_slot,
    float* __restrict__ d_out_data_f32, int* __restrict__ d_out_indices,
    double* __restrict__ group_sums, double* __restrict__ group_sq_sums,
    double* __restrict__ group_nnz, int n_target_rows, int n_cols,
    int n_groups_stats, bool compute_sq_sums, bool compute_nnz) {
    int r = blockIdx.x;
    if (r >= n_target_rows) return;
    int src_row = d_row_ids[r];
    IndptrT rs = d_indptr_full[src_row];
    IndptrT re = d_indptr_full[src_row + 1];
    int row_nnz = (int)(re - rs);
    int ds = d_out_indptr[r];
    int slot = (d_stats_codes != nullptr) ? d_stats_codes[r] : fixed_slot;
    bool accumulate = (slot >= 0 && slot < n_groups_stats);
    for (int i = threadIdx.x; i < row_nnz; i += blockDim.x) {
        InT v_in = h_data[rs + i];
        int c = (int)h_indices[rs + i];
        double v = (double)v_in;
        d_out_data_f32[ds + i] = (float)v_in;
        d_out_indices[ds + i] = c;
        if (accumulate) {
            atomicAdd(&group_sums[(size_t)slot * n_cols + c], v);
            if (compute_sq_sums) {
                atomicAdd(&group_sq_sums[(size_t)slot * n_cols + c], v * v);
            }
            if (compute_nnz && v != 0.0) {
                atomicAdd(&group_nnz[(size_t)slot * n_cols + c], 1.0);
            }
        }
    }
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
