#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_bbknn.cuh"

using namespace nb::literals;

static inline void launch_find_top_k_per_row(const float* data,
                                             const int* indptr, int n_rows,
                                             int trim, float* vals,
                                             cudaStream_t stream) {
    dim3 block(64);
    dim3 grid((n_rows + 64 - 1) / 64);
    size_t shared_mem_size =
        static_cast<size_t>(64) * static_cast<size_t>(trim) * sizeof(float);
    find_top_k_per_row_kernel<<<grid, block, shared_mem_size, stream>>>(
        data, indptr, n_rows, trim, vals);
}

static inline void launch_cut_smaller(int* indptr, int* index, float* data,
                                      float* vals, int n_rows,
                                      cudaStream_t stream) {
    dim3 grid(n_rows);
    dim3 block(64);
    cut_smaller_kernel<<<grid, block, 0, stream>>>(indptr, index, data, vals,
                                                   n_rows);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "find_top_k_per_row",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> indptr, int n_rows, int trim,
           gpu_array_c<float, Device> vals, std::uintptr_t stream) {
            launch_find_top_k_per_row(data.data(), indptr.data(), n_rows, trim,
                                      vals.data(), (cudaStream_t)stream);
        },
        "data"_a, "indptr"_a, nb::kw_only(), "n_rows"_a, "trim"_a, "vals"_a,
        "stream"_a = 0);

    m.def(
        "cut_smaller",
        [](gpu_array_c<int, Device> indptr, gpu_array_c<int, Device> index,
           gpu_array_c<float, Device> data, gpu_array_c<float, Device> vals,
           int n_rows, std::uintptr_t stream) {
            launch_cut_smaller(indptr.data(), index.data(), data.data(),
                               vals.data(), n_rows, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "vals"_a, "n_rows"_a,
        "stream"_a = 0);
}

NB_MODULE(_bbknn_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
