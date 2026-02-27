#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_normalize.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_normalize(T* X, long long rows, long long cols,
                                    cudaStream_t stream) {
    unsigned block_dim =
        std::min(256u, std::max(32u, ((unsigned)cols + 31u) / 32u * 32u));
    dim3 block(block_dim);
    dim3 grid(rows);
    normalize_kernel<T><<<grid, block, 0, stream>>>(X, rows, cols);
}

template <typename T>
static inline void launch_l2_row_normalize(const T* src, T* dst, int n_rows,
                                           int n_cols, cudaStream_t stream) {
    unsigned block_dim =
        std::min(256u, std::max(32u, ((unsigned)n_cols + 31u) / 32u * 32u));
    l2_row_normalize_kernel<T>
        <<<n_rows, block_dim, 0, stream>>>(src, dst, n_rows, n_cols);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // normalize - float32
    m.def(
        "normalize",
        [](gpu_array_c<float, Device> X, long long rows, long long cols,
           std::uintptr_t stream) {
            launch_normalize<float>(X.data(), rows, cols, (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);

    // normalize - float64
    m.def(
        "normalize",
        [](gpu_array_c<double, Device> X, long long rows, long long cols,
           std::uintptr_t stream) {
            launch_normalize<double>(X.data(), rows, cols,
                                     (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);

    // l2_row_normalize - float32
    m.def(
        "l2_row_normalize",
        [](gpu_array_c<const float, Device> src, gpu_array_c<float, Device> dst,
           int n_rows, int n_cols, std::uintptr_t stream) {
            launch_l2_row_normalize<float>(src.data(), dst.data(), n_rows,
                                           n_cols, (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "dst"_a, "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);

    // l2_row_normalize - float64
    m.def(
        "l2_row_normalize",
        [](gpu_array_c<const double, Device> src,
           gpu_array_c<double, Device> dst, int n_rows, int n_cols,
           std::uintptr_t stream) {
            launch_l2_row_normalize<double>(src.data(), dst.data(), n_rows,
                                            n_cols, (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "dst"_a, "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);
}

NB_MODULE(_harmony_normalize_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
