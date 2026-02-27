#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_outer.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_outer(T* E, const T* Pr_b, const T* R_sum,
                                long long n_cats, long long n_pcs,
                                long long switcher, cudaStream_t stream) {
    dim3 block(256);
    long long N = n_cats * n_pcs;
    dim3 grid((unsigned)((N + block.x - 1) / block.x));
    outer_kernel<T>
        <<<grid, block, 0, stream>>>(E, Pr_b, R_sum, n_cats, n_pcs, switcher);
}

template <typename T>
static inline void launch_harmony_corr(T* Z, const T* W, const int* cats,
                                       const T* R, long long n_cells,
                                       long long n_pcs, cudaStream_t stream) {
    dim3 block(256);
    long long N = n_cells * n_pcs;
    dim3 grid((unsigned)((N + block.x - 1) / block.x));
    harmony_correction_kernel<T>
        <<<grid, block, 0, stream>>>(Z, W, cats, R, n_cells, n_pcs);
}

template <typename T>
static inline void launch_batched_correction(T* Z, const T* W_all,
                                             const int* cats, const T* R,
                                             int n_cells, int n_pcs,
                                             int n_clusters, int n_batches_p1,
                                             cudaStream_t stream) {
    dim3 block(256);
    long long N = (long long)n_cells * n_pcs;
    dim3 grid((unsigned)((N + block.x - 1) / block.x));
    batched_correction_kernel<T><<<grid, block, 0, stream>>>(
        Z, W_all, cats, R, n_cells, n_pcs, n_clusters, n_batches_p1);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // outer - float32
    m.def(
        "outer",
        [](gpu_array_c<float, Device> E, gpu_array_c<const float, Device> Pr_b,
           gpu_array_c<const float, Device> R_sum, long long n_cats,
           long long n_pcs, long long switcher, std::uintptr_t stream) {
            launch_outer<float>(E.data(), Pr_b.data(), R_sum.data(), n_cats,
                                n_pcs, switcher, (cudaStream_t)stream);
        },
        "E"_a, nb::kw_only(), "Pr_b"_a, "R_sum"_a, "n_cats"_a, "n_pcs"_a,
        "switcher"_a, "stream"_a = 0);

    // outer - float64
    m.def(
        "outer",
        [](gpu_array_c<double, Device> E,
           gpu_array_c<const double, Device> Pr_b,
           gpu_array_c<const double, Device> R_sum, long long n_cats,
           long long n_pcs, long long switcher, std::uintptr_t stream) {
            launch_outer<double>(E.data(), Pr_b.data(), R_sum.data(), n_cats,
                                 n_pcs, switcher, (cudaStream_t)stream);
        },
        "E"_a, nb::kw_only(), "Pr_b"_a, "R_sum"_a, "n_cats"_a, "n_pcs"_a,
        "switcher"_a, "stream"_a = 0);

    // harmony_corr - float32
    m.def(
        "harmony_corr",
        [](gpu_array_c<float, Device> Z, gpu_array_c<const float, Device> W,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const float, Device> R, long long n_cells,
           long long n_pcs, std::uintptr_t stream) {
            launch_harmony_corr<float>(Z.data(), W.data(), cats.data(),
                                       R.data(), n_cells, n_pcs,
                                       (cudaStream_t)stream);
        },
        "Z"_a, nb::kw_only(), "W"_a, "cats"_a, "R"_a, "n_cells"_a, "n_pcs"_a,
        "stream"_a = 0);

    // harmony_corr - float64
    m.def(
        "harmony_corr",
        [](gpu_array_c<double, Device> Z, gpu_array_c<const double, Device> W,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const double, Device> R, long long n_cells,
           long long n_pcs, std::uintptr_t stream) {
            launch_harmony_corr<double>(Z.data(), W.data(), cats.data(),
                                        R.data(), n_cells, n_pcs,
                                        (cudaStream_t)stream);
        },
        "Z"_a, nb::kw_only(), "W"_a, "cats"_a, "R"_a, "n_cells"_a, "n_pcs"_a,
        "stream"_a = 0);

    // batched_correction - float32
    m.def(
        "batched_correction",
        [](gpu_array_c<float, Device> Z, gpu_array_c<const float, Device> W_all,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const float, Device> R, int n_cells, int n_pcs,
           int n_clusters, int n_batches_p1, std::uintptr_t stream) {
            launch_batched_correction<float>(
                Z.data(), W_all.data(), cats.data(), R.data(), n_cells, n_pcs,
                n_clusters, n_batches_p1, (cudaStream_t)stream);
        },
        "Z"_a, nb::kw_only(), "W_all"_a, "cats"_a, "R"_a, "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches_p1"_a, "stream"_a = 0);

    // batched_correction - float64
    m.def(
        "batched_correction",
        [](gpu_array_c<double, Device> Z,
           gpu_array_c<const double, Device> W_all,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const double, Device> R, int n_cells, int n_pcs,
           int n_clusters, int n_batches_p1, std::uintptr_t stream) {
            launch_batched_correction<double>(
                Z.data(), W_all.data(), cats.data(), R.data(), n_cells, n_pcs,
                n_clusters, n_batches_p1, (cudaStream_t)stream);
        },
        "Z"_a, nb::kw_only(), "W_all"_a, "cats"_a, "R"_a, "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches_p1"_a, "stream"_a = 0);
}

NB_MODULE(_harmony_outer_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
