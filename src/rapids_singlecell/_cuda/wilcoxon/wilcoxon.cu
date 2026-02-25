#include <cstdint>
#include <memory>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include "../nb_types.h"
#include "kernels_wilcoxon.cuh"
#include "kernels_wilcoxon_pipeline.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

static size_t get_seg_sort_temp_bytes(int n_rows, int n_cols) {
    size_t bytes = 0;
    auto* dk = reinterpret_cast<double*>(1);
    auto* dv = reinterpret_cast<int*>(1);
    auto* doff = reinterpret_cast<int*>(1);
    int n_items = n_rows * n_cols;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, bytes, dk, dk, dv, dv, n_items, n_cols, doff, doff + 1, 0, 64);
    return bytes;
}

// ============================================================================
// Fused ranking: CUB segmented sort + average rank + tie correction
// Workspace passed from caller (Python/CuPy side via _alloc_sort_workspace).
// ============================================================================

static inline void compute_ranks_impl(double* matrix, double* correction,
                                      double* sorted_vals, int* sorter,
                                      int* iota, int* offsets,
                                      uint8_t* cub_temp, size_t cub_temp_bytes,
                                      int n_rows, int n_cols,
                                      cudaStream_t stream) {
    if (n_rows == 0 || n_cols == 0) return;

    int n_items = n_rows * n_cols;
    {
        constexpr int THREADS = 256;
        dim3 iota_grid((n_rows + THREADS - 1) / THREADS, n_cols);
        iota_segments_kernel<<<iota_grid, THREADS, 0, stream>>>(iota, n_rows,
                                                                n_cols);
        int off_blocks = (n_cols + 1 + THREADS - 1) / THREADS;
        fill_offsets_kernel<<<off_blocks, THREADS, 0, stream>>>(offsets, n_rows,
                                                                n_cols);
    }

    cub::DeviceSegmentedRadixSort::SortPairs(
        cub_temp, cub_temp_bytes, matrix, sorted_vals, iota, sorter, n_items,
        n_cols, offsets, offsets + 1, 0, 64, stream);

    int threads = round_up_to_warp(n_rows);
    average_rank_kernel<<<n_cols, threads, 0, stream>>>(sorted_vals, sorter,
                                                        matrix, n_rows, n_cols);
    tie_correction_kernel<<<n_cols, threads, 0, stream>>>(
        sorted_vals, correction, n_rows, n_cols);
}

static inline void launch_tie_correction(const double* sorted_vals,
                                         double* correction, int n_rows,
                                         int n_cols, cudaStream_t stream) {
    int threads = round_up_to_warp(n_rows);
    tie_correction_kernel<<<n_cols, threads, 0, stream>>>(
        sorted_vals, correction, n_rows, n_cols);
}

static inline void launch_average_rank(const double* sorted_vals,
                                       const int* sorter, double* ranks,
                                       int n_rows, int n_cols,
                                       cudaStream_t stream) {
    int threads = round_up_to_warp(n_rows);
    average_rank_kernel<<<n_cols, threads, 0, stream>>>(sorted_vals, sorter,
                                                        ranks, n_rows, n_cols);
}

// ============================================================================
// Fill kernel
// ============================================================================

__global__ void fill_double_kernel(double* __restrict__ out, double val,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = val;
    }
}

static inline void fill_ones(double* ptr, int n, cudaStream_t stream) {
    constexpr int THREADS = 256;
    int blocks = (n + THREADS - 1) / THREADS;
    fill_double_kernel<<<blocks, THREADS, 0, stream>>>(ptr, 1.0, n);
}

// ============================================================================
// Full pipeline: CSC -> dense -> rank -> rank_sums -> z-score -> p-value
// All workspace allocated via RMM (uses whatever pool Python configured).
// ============================================================================

template <typename T>
static void wilcoxon_chunk_vs_rest_impl(
    const T* csc_data, const int* csc_indices, const int* csc_indptr,
    int n_cells, int col_start, int col_stop, const int* cell_indices,
    const int* cat_offsets, const double* group_sizes, int n_groups,
    bool tie_correct, bool use_continuity, double* z_out, double* p_out,
    cudaStream_t stream) {
    int n_genes = col_stop - col_start;
    if (n_genes == 0 || n_cells == 0) return;

    auto sv = rmm::cuda_stream_view(stream);
    auto* mr = rmm::mr::get_current_device_resource();
    size_t dense_elems = static_cast<size_t>(n_cells) * n_genes;

    rmm::device_uvector<double> dense(dense_elems, sv, mr);
    rmm::device_uvector<double> sorted_vals(dense_elems, sv, mr);
    rmm::device_uvector<int> sorter(dense_elems, sv, mr);
    rmm::device_uvector<int> iota_buf(dense_elems, sv, mr);
    rmm::device_uvector<int> offsets(n_genes + 1, sv, mr);
    rmm::device_uvector<double> correction(n_genes, sv, mr);
    rmm::device_uvector<double> rank_sums(
        static_cast<size_t>(n_groups) * n_genes, sv, mr);

    size_t cub_temp_bytes = get_seg_sort_temp_bytes(n_cells, n_genes);
    rmm::device_uvector<uint8_t> cub_temp(cub_temp_bytes, sv, mr);

    // 1. CSC slice -> dense F-order
    {
        constexpr int THREADS = 256;
        csc_slice_to_dense_kernel<T><<<n_genes, THREADS, 0, stream>>>(
            csc_data, csc_indices, csc_indptr + col_start, dense.data(),
            n_cells, n_genes);
    }

    // 2-3. Sort + rank + tie correction
    compute_ranks_impl(dense.data(), correction.data(), sorted_vals.data(),
                       sorter.data(), iota_buf.data(), offsets.data(),
                       cub_temp.data(), cub_temp_bytes, n_cells, n_genes,
                       stream);

    if (!tie_correct) {
        fill_ones(correction.data(), n_genes, stream);
    }

    // 4. Rank sums per group
    {
        int threads = round_up_to_warp(n_cells);
        dim3 grid(n_genes, n_groups);
        rank_sum_grouped_kernel<<<grid, threads, 0, stream>>>(
            dense.data(), cell_indices, cat_offsets, rank_sums.data(), n_cells,
            n_genes, n_groups);
    }

    // 5-6. Z-scores + p-values
    {
        int total = n_groups * n_genes;
        constexpr int THREADS = 256;
        int blocks = (total + THREADS - 1) / THREADS;
        zscore_pvalue_vs_rest_kernel<<<blocks, THREADS, 0, stream>>>(
            rank_sums.data(), correction.data(), group_sizes, z_out, p_out,
            n_cells, n_genes, n_groups, use_continuity);
    }
}

template <typename T>
static void wilcoxon_chunk_with_ref_impl(
    const T* csc_data, const int* csc_indices, const int* csc_indptr,
    int n_combined, int col_start, int col_stop, const bool* group_mask,
    int n_group, int n_ref, bool tie_correct, bool use_continuity,
    double* z_out, double* p_out, cudaStream_t stream) {
    int n_genes = col_stop - col_start;
    if (n_genes == 0 || n_combined == 0) return;

    auto sv = rmm::cuda_stream_view(stream);
    auto* mr = rmm::mr::get_current_device_resource();
    size_t dense_elems = static_cast<size_t>(n_combined) * n_genes;

    rmm::device_uvector<double> dense(dense_elems, sv, mr);
    rmm::device_uvector<double> sorted_vals(dense_elems, sv, mr);
    rmm::device_uvector<int> sorter(dense_elems, sv, mr);
    rmm::device_uvector<int> iota_buf(dense_elems, sv, mr);
    rmm::device_uvector<int> offsets(n_genes + 1, sv, mr);
    rmm::device_uvector<double> correction(n_genes, sv, mr);
    rmm::device_uvector<double> rank_sums(n_genes, sv, mr);

    size_t cub_temp_bytes = get_seg_sort_temp_bytes(n_combined, n_genes);
    rmm::device_uvector<uint8_t> cub_temp(cub_temp_bytes, sv, mr);

    // 1. CSC -> dense
    {
        constexpr int THREADS = 256;
        csc_slice_to_dense_kernel<T><<<n_genes, THREADS, 0, stream>>>(
            csc_data, csc_indices, csc_indptr + col_start, dense.data(),
            n_combined, n_genes);
    }

    // 2-3. Sort + rank + tie correction
    compute_ranks_impl(dense.data(), correction.data(), sorted_vals.data(),
                       sorter.data(), iota_buf.data(), offsets.data(),
                       cub_temp.data(), cub_temp_bytes, n_combined, n_genes,
                       stream);

    if (!tie_correct) {
        fill_ones(correction.data(), n_genes, stream);
    }

    // 4. Masked rank sum
    {
        int threads = round_up_to_warp(n_combined);
        rank_sum_masked_kernel<<<n_genes, threads, 0, stream>>>(
            dense.data(), group_mask, rank_sums.data(), n_combined, n_genes);
    }

    // 5-6. Z-scores + p-values
    {
        constexpr int THREADS = 256;
        int blocks = (n_genes + THREADS - 1) / THREADS;
        zscore_pvalue_with_ref_kernel<<<blocks, THREADS, 0, stream>>>(
            rank_sums.data(), correction.data(), z_out, p_out, n_combined,
            n_group, n_ref, n_genes, use_continuity);
    }
}

// ============================================================================
// RAII helper: pin host arrays for async CUDA transfers, unpin on destruction
// ============================================================================

struct HostPinner {
    std::vector<void*> ptrs;

    void pin(const void* ptr, size_t nbytes) {
        if (nbytes == 0) return;
        auto err = cudaHostRegister(const_cast<void*>(ptr), nbytes, 0);
        if (err == cudaSuccess) {
            ptrs.push_back(const_cast<void*>(ptr));
        } else {
            cudaGetLastError();  // clear error state
        }
    }

    ~HostPinner() {
        for (auto p : ptrs) cudaHostUnregister(p);
    }
};

// ============================================================================
// Host-streaming pipeline: vs-rest (pinned host → multi-GPU)
// C++ manages chunk streaming, multi-GPU dispatch, stats + ranking + z/p.
//
// 4-phase structure with per-device RMM pool:
//   Phase 1 — Create stream + pool, upload group mapping, alloc accumulators
//   Phase 2 — Process gene chunks (per-chunk allocs are pool bookkeeping)
//   Phase 3 — Sync all devices
//   Phase 4 — Cleanup (uvectors → pool → cuda_mr → stream)
// ============================================================================

template <typename T>
static void wilcoxon_vs_rest_host_impl(
    const T* h_csc_data, const int* h_csc_indices, const int64_t* h_csc_indptr,
    const int* h_cell_indices, const int* h_cat_offsets,
    const double* h_group_sizes, int n_cells, int n_groups, int n_genes,
    bool tie_correct, bool use_continuity, int chunk_width,
    const int* h_device_ids, int n_devices, double* h_z_out, double* h_p_out,
    double* h_sums_out, double* h_sq_sums_out, double* h_nnz_out) {
    if (n_genes == 0 || n_cells == 0) return;

    // Pin all host arrays for truly async cudaMemcpyAsync transfers
    int64_t nnz = h_csc_indptr[n_genes];
    int n_cell_idx = h_cat_offsets[n_groups];
    size_t out_bytes = static_cast<size_t>(n_groups) * n_genes * sizeof(double);
    HostPinner pinner;
    pinner.pin(h_csc_data, nnz * sizeof(T));
    pinner.pin(h_csc_indices, nnz * sizeof(int));
    pinner.pin(h_csc_indptr, (n_genes + 1) * sizeof(int64_t));
    pinner.pin(h_cell_indices, n_cell_idx * sizeof(int));
    pinner.pin(h_cat_offsets, (n_groups + 1) * sizeof(int));
    pinner.pin(h_group_sizes, n_groups * sizeof(double));
    pinner.pin(h_z_out, out_bytes);
    pinner.pin(h_p_out, out_bytes);
    pinner.pin(h_sums_out, out_bytes);
    pinner.pin(h_sq_sums_out, out_bytes);
    pinner.pin(h_nnz_out, out_bytes);

    using cuda_mr_t = rmm::mr::cuda_memory_resource;
    using pool_mr_t = rmm::mr::pool_memory_resource<cuda_mr_t>;

    int genes_per_device = (n_genes + n_devices - 1) / n_devices;

    struct DeviceCtx {
        cudaStream_t stream = nullptr;
        int device_id = 0;
        int g_start = 0, dev_ng = 0;
        // Destruction order: uvectors → pool → cuda_mr (reverse of declaration)
        std::unique_ptr<cuda_mr_t> cuda_mr;
        std::unique_ptr<pool_mr_t> pool;
        std::unique_ptr<rmm::device_uvector<int>> cells, cat_off;
        std::unique_ptr<rmm::device_uvector<double>> gsizes;
        std::unique_ptr<rmm::device_uvector<double>> sums, sq_sums, nnz;
    };
    std::vector<DeviceCtx> ctxs;
    ctxs.reserve(n_devices);

    // ---- Phase 1: Create pools, upload group mapping, allocate accumulators
    // --
    for (int di = 0; di < n_devices; di++) {
        int dev_id = h_device_ids[di];
        int g_start = std::min(di * genes_per_device, n_genes);
        int g_stop = std::min(g_start + genes_per_device, n_genes);
        if (g_start >= g_stop) continue;
        int dev_ng = g_stop - g_start;

        cudaSetDevice(dev_id);
        auto& ctx = ctxs.emplace_back();
        ctx.device_id = dev_id;
        ctx.g_start = g_start;
        ctx.dev_ng = dev_ng;
        cudaStreamCreate(&ctx.stream);
        auto sv = rmm::cuda_stream_view(ctx.stream);

        // Compute pool initial size: device-wide + peak per-chunk workspace
        int cg_max = std::min(chunk_width, dev_ng);
        size_t de_max = static_cast<size_t>(n_cells) * cg_max;

        int64_t max_chunk_nnz = 0;
        for (int col = g_start; col < g_stop; col += chunk_width) {
            int ce = std::min(col + chunk_width, g_stop);
            max_chunk_nnz =
                std::max(max_chunk_nnz, h_csc_indptr[ce] - h_csc_indptr[col]);
        }

        size_t pool_bytes = 0;
        // Device-wide buffers
        pool_bytes += n_cell_idx * sizeof(int);      // cells
        pool_bytes += (n_groups + 1) * sizeof(int);  // cat_off
        pool_bytes += n_groups * sizeof(double);     // gsizes
        pool_bytes += 3 * static_cast<size_t>(n_groups) * dev_ng *
                      sizeof(double);  // sums, sq_sums, nnz
        // Per-chunk CSC upload
        pool_bytes += max_chunk_nnz * (sizeof(T) + sizeof(int));
        pool_bytes += (cg_max + 1) * sizeof(int);  // indptr
        // Per-chunk workspace
        pool_bytes += de_max * 2 * sizeof(double);  // dense + sorted_v
        pool_bytes += de_max * 2 * sizeof(int);     // sorter + iota
        pool_bytes += (cg_max + 1) * sizeof(int);   // seg_off
        pool_bytes += cg_max * sizeof(double);      // corr
        pool_bytes += get_seg_sort_temp_bytes(n_cells, cg_max);
        pool_bytes += 3 * static_cast<size_t>(n_groups) * cg_max *
                      sizeof(double);  // rsums + zc + pc
        // 50% headroom for pool fragmentation + alignment padding
        pool_bytes = pool_bytes * 3 / 2;
        pool_bytes = (pool_bytes + 255) & ~size_t(255);

        ctx.cuda_mr = std::make_unique<cuda_mr_t>();
        ctx.pool = std::make_unique<pool_mr_t>(ctx.cuda_mr.get(), pool_bytes);
        auto* mr = ctx.pool.get();

        // Upload group mapping
        ctx.cells =
            std::make_unique<rmm::device_uvector<int>>(n_cell_idx, sv, mr);
        ctx.cat_off =
            std::make_unique<rmm::device_uvector<int>>(n_groups + 1, sv, mr);
        ctx.gsizes =
            std::make_unique<rmm::device_uvector<double>>(n_groups, sv, mr);

        cudaMemcpyAsync(ctx.cells->data(), h_cell_indices,
                        n_cell_idx * sizeof(int), cudaMemcpyHostToDevice,
                        ctx.stream);
        cudaMemcpyAsync(ctx.cat_off->data(), h_cat_offsets,
                        (n_groups + 1) * sizeof(int), cudaMemcpyHostToDevice,
                        ctx.stream);
        cudaMemcpyAsync(ctx.gsizes->data(), h_group_sizes,
                        n_groups * sizeof(double), cudaMemcpyHostToDevice,
                        ctx.stream);

        // Stat accumulators: (n_groups, dev_ng) row-major
        size_t dev_out = static_cast<size_t>(n_groups) * dev_ng;
        ctx.sums =
            std::make_unique<rmm::device_uvector<double>>(dev_out, sv, mr);
        ctx.sq_sums =
            std::make_unique<rmm::device_uvector<double>>(dev_out, sv, mr);
        ctx.nnz =
            std::make_unique<rmm::device_uvector<double>>(dev_out, sv, mr);
    }

    // ---- Phase 2: Process gene chunks (allocs/deallocs are pool bookkeeping)
    // -
    for (auto& ctx : ctxs) {
        cudaSetDevice(ctx.device_id);
        auto sv = rmm::cuda_stream_view(ctx.stream);
        auto* mr = ctx.pool.get();
        int g_stop = ctx.g_start + ctx.dev_ng;

        for (int col_start = ctx.g_start; col_start < g_stop;
             col_start += chunk_width) {
            int col_stop = std::min(col_start + chunk_width, g_stop);
            int cg = col_stop - col_start;
            int gene_off = col_start - ctx.g_start;

            int64_t nnz_s = h_csc_indptr[col_start];
            int64_t nnz_e = h_csc_indptr[col_stop];
            int64_t chunk_nnz = nnz_e - nnz_s;

            // H2D: CSC slice (pool alloc)
            rmm::device_uvector<T> d_data(chunk_nnz, sv, mr);
            rmm::device_uvector<int> d_indices(chunk_nnz, sv, mr);
            rmm::device_uvector<int> d_indptr(cg + 1, sv, mr);

            if (chunk_nnz > 0) {
                cudaMemcpyAsync(d_data.data(), h_csc_data + nnz_s,
                                chunk_nnz * sizeof(T), cudaMemcpyHostToDevice,
                                ctx.stream);
                cudaMemcpyAsync(d_indices.data(), h_csc_indices + nnz_s,
                                chunk_nnz * sizeof(int), cudaMemcpyHostToDevice,
                                ctx.stream);
            }

            std::vector<int> adj(cg + 1);
            for (int i = 0; i <= cg; i++)
                adj[i] = static_cast<int>(h_csc_indptr[col_start + i] - nnz_s);
            cudaMemcpyAsync(d_indptr.data(), adj.data(), (cg + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, ctx.stream);

            // Per-chunk workspace (pool alloc — returned to pool at scope exit)
            size_t de = static_cast<size_t>(n_cells) * cg;
            rmm::device_uvector<double> dense(de, sv, mr);
            rmm::device_uvector<double> sorted_v(de, sv, mr);
            rmm::device_uvector<int> sorter(de, sv, mr);
            rmm::device_uvector<int> iota(de, sv, mr);
            rmm::device_uvector<int> seg_off(cg + 1, sv, mr);
            rmm::device_uvector<double> corr(cg, sv, mr);
            size_t cub_bytes = get_seg_sort_temp_bytes(n_cells, cg);
            rmm::device_uvector<uint8_t> cub_tmp(cub_bytes, sv, mr);
            size_t grp_g = static_cast<size_t>(n_groups) * cg;
            rmm::device_uvector<double> rsums(grp_g, sv, mr);
            rmm::device_uvector<double> zc(grp_g, sv, mr);
            rmm::device_uvector<double> pc(grp_g, sv, mr);

            csc_slice_to_dense_kernel<T><<<cg, 256, 0, ctx.stream>>>(
                d_data.data(), d_indices.data(), d_indptr.data(), dense.data(),
                n_cells, cg);

            {
                int thr = round_up_to_warp(n_cells);
                stats_grouped_kernel<<<dim3(cg, n_groups), thr, 0,
                                       ctx.stream>>>(
                    dense.data(), ctx.cells->data(), ctx.cat_off->data(),
                    ctx.sums->data(), ctx.sq_sums->data(), ctx.nnz->data(),
                    n_cells, cg, n_groups, gene_off, ctx.dev_ng);
            }

            compute_ranks_impl(dense.data(), corr.data(), sorted_v.data(),
                               sorter.data(), iota.data(), seg_off.data(),
                               cub_tmp.data(), cub_bytes, n_cells, cg,
                               ctx.stream);
            if (!tie_correct) fill_ones(corr.data(), cg, ctx.stream);

            {
                int thr = round_up_to_warp(n_cells);
                rank_sum_grouped_kernel<<<dim3(cg, n_groups), thr, 0,
                                          ctx.stream>>>(
                    dense.data(), ctx.cells->data(), ctx.cat_off->data(),
                    rsums.data(), n_cells, cg, n_groups);
            }

            {
                int total = n_groups * cg;
                int blk = (total + 255) / 256;
                zscore_pvalue_vs_rest_kernel<<<blk, 256, 0, ctx.stream>>>(
                    rsums.data(), corr.data(), ctx.gsizes->data(), zc.data(),
                    pc.data(), n_cells, cg, n_groups, use_continuity);
            }

            cudaMemcpy2DAsync(h_z_out + col_start, n_genes * sizeof(double),
                              zc.data(), cg * sizeof(double),
                              cg * sizeof(double), n_groups,
                              cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpy2DAsync(h_p_out + col_start, n_genes * sizeof(double),
                              pc.data(), cg * sizeof(double),
                              cg * sizeof(double), n_groups,
                              cudaMemcpyDeviceToHost, ctx.stream);
        }

        // D2H: device-wide stats (queued on stream before moving to next
        // device)
        cudaMemcpy2DAsync(h_sums_out + ctx.g_start, n_genes * sizeof(double),
                          ctx.sums->data(), ctx.dev_ng * sizeof(double),
                          ctx.dev_ng * sizeof(double), n_groups,
                          cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpy2DAsync(h_sq_sums_out + ctx.g_start, n_genes * sizeof(double),
                          ctx.sq_sums->data(), ctx.dev_ng * sizeof(double),
                          ctx.dev_ng * sizeof(double), n_groups,
                          cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpy2DAsync(h_nnz_out + ctx.g_start, n_genes * sizeof(double),
                          ctx.nnz->data(), ctx.dev_ng * sizeof(double),
                          ctx.dev_ng * sizeof(double), n_groups,
                          cudaMemcpyDeviceToHost, ctx.stream);
    }

    // ---- Phase 3: Sync all devices ----
    for (auto& ctx : ctxs) {
        cudaSetDevice(ctx.device_id);
        cudaStreamSynchronize(ctx.stream);
    }

    // ---- Phase 4: Cleanup (uvectors → pool → cuda_mr → stream) ----
    for (auto& ctx : ctxs) {
        cudaSetDevice(ctx.device_id);
        ctx.cells.reset();
        ctx.cat_off.reset();
        ctx.gsizes.reset();
        ctx.sums.reset();
        ctx.sq_sums.reset();
        ctx.nnz.reset();
        ctx.pool.reset();
        ctx.cuda_mr.reset();
        cudaStreamDestroy(ctx.stream);
    }
}

// ============================================================================
// Host-streaming pipeline: with-reference (pinned host → single GPU)
// Same pool pattern as vs-rest but single device.
// ============================================================================

template <typename T>
static void wilcoxon_with_ref_host_impl(
    const T* h_csc_data, const int* h_csc_indices, const int64_t* h_csc_indptr,
    const int* h_row_map,      // (n_total_cells,): old → new or -1
    const bool* h_group_mask,  // (n_combined,)
    int n_total_cells, int n_combined, int n_group, int n_ref, int n_genes,
    bool tie_correct, bool use_continuity, int chunk_width, double* h_z_out,
    double* h_p_out, double* h_group_sums, double* h_group_sq_sums,
    double* h_group_nnz, double* h_ref_sums, double* h_ref_sq_sums,
    double* h_ref_nnz) {
    if (n_genes == 0 || n_combined == 0) return;

    // Pin all host arrays for truly async cudaMemcpyAsync transfers
    int64_t nnz = h_csc_indptr[n_genes];
    HostPinner pinner;
    pinner.pin(h_csc_data, nnz * sizeof(T));
    pinner.pin(h_csc_indices, nnz * sizeof(int));
    pinner.pin(h_csc_indptr, (n_genes + 1) * sizeof(int64_t));
    pinner.pin(h_row_map, n_total_cells * sizeof(int));
    pinner.pin(h_group_mask, n_combined * sizeof(bool));
    pinner.pin(h_z_out, n_genes * sizeof(double));
    pinner.pin(h_p_out, n_genes * sizeof(double));
    pinner.pin(h_group_sums, n_genes * sizeof(double));
    pinner.pin(h_group_sq_sums, n_genes * sizeof(double));
    pinner.pin(h_group_nnz, n_genes * sizeof(double));
    pinner.pin(h_ref_sums, n_genes * sizeof(double));
    pinner.pin(h_ref_sq_sums, n_genes * sizeof(double));
    pinner.pin(h_ref_nnz, n_genes * sizeof(double));

    using cuda_mr_t = rmm::mr::cuda_memory_resource;
    using pool_mr_t = rmm::mr::pool_memory_resource<cuda_mr_t>;

    // Stream created outside scope block so it outlives pool + uvectors
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {  // Scope block: all pool + uvectors destroyed here (stream still alive)
        auto sv = rmm::cuda_stream_view(stream);

        // Compute pool initial size
        int cg_max = std::min(chunk_width, n_genes);
        size_t de_max = static_cast<size_t>(n_combined) * cg_max;

        int64_t max_chunk_nnz = 0;
        for (int col = 0; col < n_genes; col += chunk_width) {
            int ce = std::min(col + chunk_width, n_genes);
            max_chunk_nnz =
                std::max(max_chunk_nnz, h_csc_indptr[ce] - h_csc_indptr[col]);
        }

        size_t pool_bytes = 0;
        // Device-wide buffers
        pool_bytes += n_total_cells * sizeof(int);  // row_map
        pool_bytes += n_combined * sizeof(bool);    // group_mask
        pool_bytes += n_combined * sizeof(int);     // stat_cells
        pool_bytes += 3 * sizeof(int);              // stat_off
        pool_bytes += 3 * 2 * static_cast<size_t>(n_genes) *
                      sizeof(double);  // sums, sq_sums, nnz
        // Per-chunk CSC upload
        pool_bytes += max_chunk_nnz * (sizeof(T) + sizeof(int));
        pool_bytes += (cg_max + 1) * sizeof(int);  // indptr
        // Per-chunk workspace
        pool_bytes += de_max * 2 * sizeof(double);  // dense + sorted_v
        pool_bytes += de_max * 2 * sizeof(int);     // sorter + iota
        pool_bytes += (cg_max + 1) * sizeof(int);   // seg_off
        pool_bytes += cg_max * sizeof(double);      // corr
        pool_bytes += get_seg_sort_temp_bytes(n_combined, cg_max);
        pool_bytes += 3 * cg_max * sizeof(double);  // rsums + zc + pc
        pool_bytes = pool_bytes * 3 / 2;
        pool_bytes = (pool_bytes + 255) & ~size_t(255);

        cuda_mr_t cuda_mr;
        pool_mr_t pool(&cuda_mr, pool_bytes);
        auto* mr = &pool;

        // Upload row_map and group_mask
        rmm::device_uvector<int> d_row_map(n_total_cells, sv, mr);
        rmm::device_uvector<bool> d_group_mask(n_combined, sv, mr);
        cudaMemcpyAsync(d_row_map.data(), h_row_map,
                        n_total_cells * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(d_group_mask.data(), h_group_mask,
                        n_combined * sizeof(bool), cudaMemcpyHostToDevice,
                        stream);

        // Build 2-group cell mapping from group_mask (for stats kernel)
        std::vector<int> filt_cells;
        filt_cells.reserve(n_combined);
        for (int i = 0; i < n_combined; i++)
            if (h_group_mask[i]) filt_cells.push_back(i);
        int grp_end = static_cast<int>(filt_cells.size());
        for (int i = 0; i < n_combined; i++)
            if (!h_group_mask[i]) filt_cells.push_back(i);
        int filt_offsets[3] = {0, grp_end, static_cast<int>(filt_cells.size())};

        rmm::device_uvector<int> d_stat_cells(n_combined, sv, mr);
        rmm::device_uvector<int> d_stat_off(3, sv, mr);
        cudaMemcpyAsync(d_stat_cells.data(), filt_cells.data(),
                        n_combined * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(d_stat_off.data(), filt_offsets, 3 * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        // Stat accumulators: 2 groups × n_genes (row 0 = group, row 1 = ref)
        size_t stat_elems = 2 * static_cast<size_t>(n_genes);
        rmm::device_uvector<double> d_sums(stat_elems, sv, mr);
        rmm::device_uvector<double> d_sq_sums(stat_elems, sv, mr);
        rmm::device_uvector<double> d_nnz_stat(stat_elems, sv, mr);

        // ---- Process gene chunks ----
        for (int col_start = 0; col_start < n_genes; col_start += chunk_width) {
            int col_stop = std::min(col_start + chunk_width, n_genes);
            int cg = col_stop - col_start;

            int64_t nnz_s = h_csc_indptr[col_start];
            int64_t nnz_e = h_csc_indptr[col_stop];
            int64_t chunk_nnz = nnz_e - nnz_s;

            // H2D: CSC slice (pool alloc)
            rmm::device_uvector<T> d_data(chunk_nnz, sv, mr);
            rmm::device_uvector<int> d_indices(chunk_nnz, sv, mr);
            rmm::device_uvector<int> d_indptr(cg + 1, sv, mr);

            if (chunk_nnz > 0) {
                cudaMemcpyAsync(d_data.data(), h_csc_data + nnz_s,
                                chunk_nnz * sizeof(T), cudaMemcpyHostToDevice,
                                stream);
                cudaMemcpyAsync(d_indices.data(), h_csc_indices + nnz_s,
                                chunk_nnz * sizeof(int), cudaMemcpyHostToDevice,
                                stream);
            }

            std::vector<int> adj(cg + 1);
            for (int i = 0; i <= cg; i++)
                adj[i] = static_cast<int>(h_csc_indptr[col_start + i] - nnz_s);
            cudaMemcpyAsync(d_indptr.data(), adj.data(), (cg + 1) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);

            // Per-chunk workspace (pool alloc — returned to pool at scope exit)
            size_t de = static_cast<size_t>(n_combined) * cg;
            rmm::device_uvector<double> dense(de, sv, mr);
            rmm::device_uvector<double> sorted_v(de, sv, mr);
            rmm::device_uvector<int> sorter(de, sv, mr);
            rmm::device_uvector<int> iota(de, sv, mr);
            rmm::device_uvector<int> seg_off(cg + 1, sv, mr);
            rmm::device_uvector<double> corr(cg, sv, mr);
            size_t cub_bytes = get_seg_sort_temp_bytes(n_combined, cg);
            rmm::device_uvector<uint8_t> cub_tmp(cub_bytes, sv, mr);
            rmm::device_uvector<double> rsums(cg, sv, mr);
            rmm::device_uvector<double> zc(cg, sv, mr);
            rmm::device_uvector<double> pc(cg, sv, mr);

            // CSC → filtered dense (n_combined rows via row_map)
            csc_slice_to_dense_filtered_kernel<T><<<cg, 256, 0, stream>>>(
                d_data.data(), d_indices.data(), d_indptr.data(),
                d_row_map.data(), dense.data(), n_combined, cg);

            // Stats from filtered dense (before ranking)
            {
                int thr = round_up_to_warp(n_combined);
                constexpr int N_STAT_GROUPS = 2;
                stats_grouped_kernel<<<dim3(cg, N_STAT_GROUPS), thr, 0,
                                       stream>>>(
                    dense.data(), d_stat_cells.data(), d_stat_off.data(),
                    d_sums.data(), d_sq_sums.data(), d_nnz_stat.data(),
                    n_combined, cg, N_STAT_GROUPS, col_start, n_genes);
            }

            // Sort + rank + tie correction
            compute_ranks_impl(dense.data(), corr.data(), sorted_v.data(),
                               sorter.data(), iota.data(), seg_off.data(),
                               cub_tmp.data(), cub_bytes, n_combined, cg,
                               stream);
            if (!tie_correct) fill_ones(corr.data(), cg, stream);

            // Masked rank sum (group cells only)
            {
                int thr = round_up_to_warp(n_combined);
                rank_sum_masked_kernel<<<cg, thr, 0, stream>>>(
                    dense.data(), d_group_mask.data(), rsums.data(), n_combined,
                    cg);
            }

            // Z-scores + p-values
            {
                int blk = (cg + 255) / 256;
                zscore_pvalue_with_ref_kernel<<<blk, 256, 0, stream>>>(
                    rsums.data(), corr.data(), zc.data(), pc.data(), n_combined,
                    n_group, n_ref, cg, use_continuity);
            }

            // D2H: z/p for this chunk
            cudaMemcpyAsync(h_z_out + col_start, zc.data(), cg * sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_p_out + col_start, pc.data(), cg * sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
        }

        // D2H: stats — row 0 = group, row 1 = ref
        cudaMemcpyAsync(h_group_sums, d_sums.data(), n_genes * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_ref_sums, d_sums.data() + n_genes,
                        n_genes * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(h_group_sq_sums, d_sq_sums.data(),
                        n_genes * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(h_ref_sq_sums, d_sq_sums.data() + n_genes,
                        n_genes * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(h_group_nnz, d_nnz_stat.data(),
                        n_genes * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(h_ref_nnz, d_nnz_stat.data() + n_genes,
                        n_genes * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);

        cudaStreamSynchronize(stream);
    }  // Scope exit: uvectors → pool → cuda_mr destroyed (stream still alive)

    cudaStreamDestroy(stream);
}

// ============================================================================
// Nanobind module
// ============================================================================

NB_MODULE(_wilcoxon_cuda, m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test";

    m.def("get_sort_temp_bytes", &get_seg_sort_temp_bytes, "n_rows"_a,
          "n_cols"_a);

    // Fused ranking (workspace passed from Python)
    m.def(
        "compute_ranks",
        [](cuda_array_f<double> matrix, cuda_array<double> correction,
           cuda_array_f<double> sorted_vals, cuda_array_f<int> sorter,
           cuda_array_f<int> iota, cuda_array<int> offsets,
           cuda_array<uint8_t> cub_temp, int n_rows, int n_cols,
           std::uintptr_t stream) {
            compute_ranks_impl(matrix.data(), correction.data(),
                               sorted_vals.data(), sorter.data(), iota.data(),
                               offsets.data(), cub_temp.data(), cub_temp.size(),
                               n_rows, n_cols, (cudaStream_t)stream);
        },
        "matrix"_a, "correction"_a, "sorted_vals"_a, "sorter"_a, "iota"_a,
        "offsets"_a, "cub_temp"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);

    m.def(
        "tie_correction",
        [](cuda_array_f<const double> sorted_vals,
           cuda_array<double> correction, int n_rows, int n_cols,
           std::uintptr_t stream) {
            launch_tie_correction(sorted_vals.data(), correction.data(), n_rows,
                                  n_cols, (cudaStream_t)stream);
        },
        "sorted_vals"_a, "correction"_a, nb::kw_only(), "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);

    m.def(
        "average_rank",
        [](cuda_array_f<const double> sorted_vals,
           cuda_array_f<const int> sorter, cuda_array_f<double> ranks,
           int n_rows, int n_cols, std::uintptr_t stream) {
            launch_average_rank(sorted_vals.data(), sorter.data(), ranks.data(),
                                n_rows, n_cols, (cudaStream_t)stream);
        },
        "sorted_vals"_a, "sorter"_a, "ranks"_a, nb::kw_only(), "n_rows"_a,
        "n_cols"_a, "stream"_a = 0);

    // ========================================================================
    // Full pipeline: vs-rest (workspace via RMM internally)
    // ========================================================================
    m.def(
        "wilcoxon_chunk_vs_rest",
        [](cuda_array<const float> csc_data, cuda_array<const int> csc_indices,
           cuda_array<const int> csc_indptr, int n_cells, int col_start,
           int col_stop, cuda_array<const int> cell_indices,
           cuda_array<const int> cat_offsets,
           cuda_array<const double> group_sizes, int n_groups, bool tie_correct,
           bool use_continuity, cuda_array<double> z_out,
           cuda_array<double> p_out, std::uintptr_t stream) {
            wilcoxon_chunk_vs_rest_impl<float>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(), n_cells,
                col_start, col_stop, cell_indices.data(), cat_offsets.data(),
                group_sizes.data(), n_groups, tie_correct, use_continuity,
                z_out.data(), p_out.data(), (cudaStream_t)stream);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "n_cells"_a,
        "col_start"_a, "col_stop"_a, "cell_indices"_a, "cat_offsets"_a,
        "group_sizes"_a, "n_groups"_a, "tie_correct"_a, "use_continuity"_a,
        "z_out"_a, "p_out"_a, nb::kw_only(), "stream"_a = 0);

    m.def(
        "wilcoxon_chunk_vs_rest",
        [](cuda_array<const double> csc_data, cuda_array<const int> csc_indices,
           cuda_array<const int> csc_indptr, int n_cells, int col_start,
           int col_stop, cuda_array<const int> cell_indices,
           cuda_array<const int> cat_offsets,
           cuda_array<const double> group_sizes, int n_groups, bool tie_correct,
           bool use_continuity, cuda_array<double> z_out,
           cuda_array<double> p_out, std::uintptr_t stream) {
            wilcoxon_chunk_vs_rest_impl<double>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(), n_cells,
                col_start, col_stop, cell_indices.data(), cat_offsets.data(),
                group_sizes.data(), n_groups, tie_correct, use_continuity,
                z_out.data(), p_out.data(), (cudaStream_t)stream);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "n_cells"_a,
        "col_start"_a, "col_stop"_a, "cell_indices"_a, "cat_offsets"_a,
        "group_sizes"_a, "n_groups"_a, "tie_correct"_a, "use_continuity"_a,
        "z_out"_a, "p_out"_a, nb::kw_only(), "stream"_a = 0);

    // ========================================================================
    // Full pipeline: with-reference (workspace via RMM internally)
    // ========================================================================
    m.def(
        "wilcoxon_chunk_with_ref",
        [](cuda_array<const float> csc_data, cuda_array<const int> csc_indices,
           cuda_array<const int> csc_indptr, int n_combined, int col_start,
           int col_stop, cuda_array<const bool> group_mask, int n_group,
           int n_ref, bool tie_correct, bool use_continuity,
           cuda_array<double> z_out, cuda_array<double> p_out,
           std::uintptr_t stream) {
            wilcoxon_chunk_with_ref_impl<float>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                n_combined, col_start, col_stop, group_mask.data(), n_group,
                n_ref, tie_correct, use_continuity, z_out.data(), p_out.data(),
                (cudaStream_t)stream);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "n_combined"_a,
        "col_start"_a, "col_stop"_a, "group_mask"_a, "n_group"_a, "n_ref"_a,
        "tie_correct"_a, "use_continuity"_a, "z_out"_a, "p_out"_a,
        nb::kw_only(), "stream"_a = 0);

    m.def(
        "wilcoxon_chunk_with_ref",
        [](cuda_array<const double> csc_data, cuda_array<const int> csc_indices,
           cuda_array<const int> csc_indptr, int n_combined, int col_start,
           int col_stop, cuda_array<const bool> group_mask, int n_group,
           int n_ref, bool tie_correct, bool use_continuity,
           cuda_array<double> z_out, cuda_array<double> p_out,
           std::uintptr_t stream) {
            wilcoxon_chunk_with_ref_impl<double>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                n_combined, col_start, col_stop, group_mask.data(), n_group,
                n_ref, tie_correct, use_continuity, z_out.data(), p_out.data(),
                (cudaStream_t)stream);
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "n_combined"_a,
        "col_start"_a, "col_stop"_a, "group_mask"_a, "n_group"_a, "n_ref"_a,
        "tie_correct"_a, "use_continuity"_a, "z_out"_a, "p_out"_a,
        nb::kw_only(), "stream"_a = 0);

    // ========================================================================
    // Host-streaming pipeline: vs-rest (pinned host → multi-GPU)
    // ========================================================================
    m.def(
        "wilcoxon_vs_rest_host",
        [](host_array<const float> csc_data, host_array<const int> csc_indices,
           host_array<const int64_t> csc_indptr,
           host_array<const int> cell_indices,
           host_array<const int> cat_offsets,
           host_array<const double> group_sizes, int n_cells, int n_groups,
           int n_genes, bool tie_correct, bool use_continuity, int chunk_width,
           host_array<const int> device_ids, host_array<double> z_out,
           host_array<double> p_out, host_array<double> sums_out,
           host_array<double> sq_sums_out, host_array<double> nnz_out) {
            wilcoxon_vs_rest_host_impl<float>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                cell_indices.data(), cat_offsets.data(), group_sizes.data(),
                n_cells, n_groups, n_genes, tie_correct, use_continuity,
                chunk_width, device_ids.data(),
                static_cast<int>(device_ids.size()), z_out.data(), p_out.data(),
                sums_out.data(), sq_sums_out.data(), nnz_out.data());
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "cell_indices"_a,
        "cat_offsets"_a, "group_sizes"_a, "n_cells"_a, "n_groups"_a,
        "n_genes"_a, "tie_correct"_a, "use_continuity"_a, "chunk_width"_a,
        "device_ids"_a, "z_out"_a, "p_out"_a, "sums_out"_a, "sq_sums_out"_a,
        "nnz_out"_a);

    m.def(
        "wilcoxon_vs_rest_host",
        [](host_array<const double> csc_data, host_array<const int> csc_indices,
           host_array<const int64_t> csc_indptr,
           host_array<const int> cell_indices,
           host_array<const int> cat_offsets,
           host_array<const double> group_sizes, int n_cells, int n_groups,
           int n_genes, bool tie_correct, bool use_continuity, int chunk_width,
           host_array<const int> device_ids, host_array<double> z_out,
           host_array<double> p_out, host_array<double> sums_out,
           host_array<double> sq_sums_out, host_array<double> nnz_out) {
            wilcoxon_vs_rest_host_impl<double>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                cell_indices.data(), cat_offsets.data(), group_sizes.data(),
                n_cells, n_groups, n_genes, tie_correct, use_continuity,
                chunk_width, device_ids.data(),
                static_cast<int>(device_ids.size()), z_out.data(), p_out.data(),
                sums_out.data(), sq_sums_out.data(), nnz_out.data());
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "cell_indices"_a,
        "cat_offsets"_a, "group_sizes"_a, "n_cells"_a, "n_groups"_a,
        "n_genes"_a, "tie_correct"_a, "use_continuity"_a, "chunk_width"_a,
        "device_ids"_a, "z_out"_a, "p_out"_a, "sums_out"_a, "sq_sums_out"_a,
        "nnz_out"_a);

    // ========================================================================
    // Host-streaming pipeline: with-reference (pinned host → single GPU)
    // ========================================================================
    m.def(
        "wilcoxon_with_ref_host",
        [](host_array<const float> csc_data, host_array<const int> csc_indices,
           host_array<const int64_t> csc_indptr, host_array<const int> row_map,
           host_array<const bool> group_mask, int n_total_cells, int n_combined,
           int n_group, int n_ref, int n_genes, bool tie_correct,
           bool use_continuity, int chunk_width, host_array<double> z_out,
           host_array<double> p_out, host_array<double> group_sums,
           host_array<double> group_sq_sums, host_array<double> group_nnz,
           host_array<double> ref_sums, host_array<double> ref_sq_sums,
           host_array<double> ref_nnz) {
            wilcoxon_with_ref_host_impl<float>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                row_map.data(), group_mask.data(), n_total_cells, n_combined,
                n_group, n_ref, n_genes, tie_correct, use_continuity,
                chunk_width, z_out.data(), p_out.data(), group_sums.data(),
                group_sq_sums.data(), group_nnz.data(), ref_sums.data(),
                ref_sq_sums.data(), ref_nnz.data());
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "row_map"_a,
        "group_mask"_a, "n_total_cells"_a, "n_combined"_a, "n_group"_a,
        "n_ref"_a, "n_genes"_a, "tie_correct"_a, "use_continuity"_a,
        "chunk_width"_a, "z_out"_a, "p_out"_a, "group_sums"_a,
        "group_sq_sums"_a, "group_nnz"_a, "ref_sums"_a, "ref_sq_sums"_a,
        "ref_nnz"_a);

    m.def(
        "wilcoxon_with_ref_host",
        [](host_array<const double> csc_data, host_array<const int> csc_indices,
           host_array<const int64_t> csc_indptr, host_array<const int> row_map,
           host_array<const bool> group_mask, int n_total_cells, int n_combined,
           int n_group, int n_ref, int n_genes, bool tie_correct,
           bool use_continuity, int chunk_width, host_array<double> z_out,
           host_array<double> p_out, host_array<double> group_sums,
           host_array<double> group_sq_sums, host_array<double> group_nnz,
           host_array<double> ref_sums, host_array<double> ref_sq_sums,
           host_array<double> ref_nnz) {
            wilcoxon_with_ref_host_impl<double>(
                csc_data.data(), csc_indices.data(), csc_indptr.data(),
                row_map.data(), group_mask.data(), n_total_cells, n_combined,
                n_group, n_ref, n_genes, tie_correct, use_continuity,
                chunk_width, z_out.data(), p_out.data(), group_sums.data(),
                group_sq_sums.data(), group_nnz.data(), ref_sums.data(),
                ref_sq_sums.data(), ref_nnz.data());
        },
        "csc_data"_a, "csc_indices"_a, "csc_indptr"_a, "row_map"_a,
        "group_mask"_a, "n_total_cells"_a, "n_combined"_a, "n_group"_a,
        "n_ref"_a, "n_genes"_a, "tie_correct"_a, "use_continuity"_a,
        "chunk_width"_a, "z_out"_a, "p_out"_a, "group_sums"_a,
        "group_sq_sums"_a, "group_nnz"_a, "ref_sums"_a, "ref_sq_sums"_a,
        "ref_nnz"_a);
}
