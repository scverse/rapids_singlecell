#include "kernels_kde.cuh"
#include "../nb_types.h"

using namespace nb::literals;

template <typename T>
inline void launch_gaussian_kde_2d(const T* xy, T* out, int n, T neg_inv_2h2,
                                   cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    gaussian_kde_2d_kernel<<<blocks, threads, 0, stream>>>(xy, out, n,
                                                           neg_inv_2h2);
}

NB_MODULE(_kde_cuda, m) {
    m.def(
        "gaussian_kde_2d",
        [](cuda_array_c<const float> xy, cuda_array_c<float> out, int n,
           float neg_inv_2h2, std::uintptr_t stream) {
            launch_gaussian_kde_2d(xy.data(), out.data(), n, neg_inv_2h2,
                                   (cudaStream_t)stream);
        },
        "xy"_a, nb::kw_only(), "out"_a, "n"_a, "neg_inv_2h2"_a, "stream"_a = 0);

    m.def(
        "gaussian_kde_2d",
        [](cuda_array_c<const double> xy, cuda_array_c<double> out, int n,
           double neg_inv_2h2, std::uintptr_t stream) {
            launch_gaussian_kde_2d(xy.data(), out.data(), n, neg_inv_2h2,
                                   (cudaStream_t)stream);
        },
        "xy"_a, nb::kw_only(), "out"_a, "n"_a, "neg_inv_2h2"_a, "stream"_a = 0);
}
