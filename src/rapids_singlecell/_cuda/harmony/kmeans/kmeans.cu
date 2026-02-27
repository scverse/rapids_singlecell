#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_kmeans.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_kmeans_err(const T* r, const T* dot, size_t n, T* out,
                                     cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int threads = 256;
    int blocks = std::min((int)((n + threads - 1) / threads),
                          prop.multiProcessorCount * 8);
    kmeans_err_kernel<T><<<blocks, threads, 0, stream>>>(r, dot, n, out);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // -- Test-only bindings below --
    // kmeans_err is used internally by compute_objective in the C++ clustering
    // loop. The binding exists solely for unit testing.

    // kmeans_err - float32
    m.def(
        "kmeans_err",
        [](gpu_array_c<const float, Device> r,
           gpu_array_c<const float, Device> dot, size_t n,
           gpu_array_c<float, Device> out, std::uintptr_t stream) {
            launch_kmeans_err<float>(r.data(), dot.data(), n, out.data(),
                                     (cudaStream_t)stream);
        },
        "r"_a, nb::kw_only(), "dot"_a, "n"_a, "out"_a, "stream"_a = 0);

    // kmeans_err - float64
    m.def(
        "kmeans_err",
        [](gpu_array_c<const double, Device> r,
           gpu_array_c<const double, Device> dot, size_t n,
           gpu_array_c<double, Device> out, std::uintptr_t stream) {
            launch_kmeans_err<double>(r.data(), dot.data(), n, out.data(),
                                      (cudaStream_t)stream);
        },
        "r"_a, nb::kw_only(), "dot"_a, "n"_a, "out"_a, "stream"_a = 0);
}

NB_MODULE(_harmony_kmeans_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
