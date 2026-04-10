#include <cstdint>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "kernels_wilcoxon_ovo.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

static size_t get_seg_sort_temp_bytes(int n_items, int n_segments) {
    size_t bytes = 0;
    auto* dk = reinterpret_cast<float*>(1);
    auto* doff = reinterpret_cast<int*>(1);
    cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes, dk, dk, n_items,
                                            n_segments, doff, doff + 1, 0, 32);
    return bytes;
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test";

    m.def("get_seg_sort_temp_bytes", &get_seg_sort_temp_bytes, "n_items"_a,
          "n_segments"_a);

    m.def(
        "segmented_sort",
        [](gpu_array_c<const float, Device> keys_in,
           gpu_array_c<float, Device> keys_out,
           gpu_array_c<const int, Device> offsets,
           gpu_array_c<uint8_t, Device> cub_temp, int n_items, int n_segments,
           std::uintptr_t stream) {
            size_t temp_bytes = cub_temp.size();
            cub::DeviceSegmentedRadixSort::SortKeys(
                cub_temp.data(), temp_bytes, keys_in.data(), keys_out.data(),
                n_items, n_segments, offsets.data(), offsets.data() + 1, 0, 32,
                (cudaStream_t)stream);
            CUDA_CHECK_LAST_ERROR(DeviceSegmentedRadixSort);
        },
        "keys_in"_a, "keys_out"_a, "offsets"_a, "cub_temp"_a, nb::kw_only(),
        "n_items"_a, "n_segments"_a, "stream"_a = 0);

    m.def(
        "csr_extract_dense",
        [](gpu_array_c<const double, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> row_ids,
           gpu_array_f<double, Device> out, int n_target, int col_start,
           int col_stop, std::uintptr_t stream) {
            int tpb = round_up_to_warp(n_target);
            int blocks = (n_target + tpb - 1) / tpb;
            csr_extract_dense_kernel<<<blocks, tpb, 0, (cudaStream_t)stream>>>(
                data.data(), indices.data(), indptr.data(), row_ids.data(),
                out.data(), n_target, col_start, col_stop);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);
        },
        "data"_a, "indices"_a, "indptr"_a, "row_ids"_a, "out"_a, nb::kw_only(),
        "n_target"_a, "col_start"_a, "col_stop"_a, "stream"_a = 0);

    m.def(
        "grouped_stats",
        [](gpu_array_f<const double, Device> data,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> sums,
           gpu_array_c<double, Device> sq_sums,
           gpu_array_c<double, Device> nnz_counts, int n_all_rows, int n_cols,
           int n_groups, bool compute_nnz, std::uintptr_t stream) {
            constexpr int THREADS = 256;
            int smem = 3 * n_groups * sizeof(double);
            grouped_stats_kernel<<<n_cols, THREADS, smem,
                                   (cudaStream_t)stream>>>(
                data.data(), grp_offsets.data(), sums.data(), sq_sums.data(),
                nnz_counts.data(), n_all_rows, n_cols, n_groups, compute_nnz);
            CUDA_CHECK_LAST_ERROR(grouped_stats_kernel);
        },
        "data"_a, "grp_offsets"_a, "sums"_a, "sq_sums"_a, "nnz_counts"_a,
        nb::kw_only(), "n_all_rows"_a, "n_cols"_a, "n_groups"_a,
        "compute_nnz"_a, "stream"_a = 0);
}

NB_MODULE(_wilcoxon_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
