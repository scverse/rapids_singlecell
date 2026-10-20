#include <cuda_runtime.h>

#include "../nb_types.h"

#include "kernels_guide_assignment.cuh"

using namespace nb::literals;

static inline void launch_fit_poisson_gauss(
    const float* values, const float* weights, const float* log_gamma,
    const float* inits, float* params, float* loss, int n_guides, int ld,
    int n_values, int n_starts, int n_iter, float tol, cudaStream_t stream) {
    if (n_guides == 0 || n_starts == 0) return;

    dim3 block(guide_assignment::BLOCK_SIZE);
    dim3 grid(n_guides, n_starts);
    guide_assignment::fit_poisson_gauss_kernel<<<grid, block, 0, stream>>>(
        values, weights, log_gamma, inits, params, loss, ld, n_values, n_iter,
        tol);
    CUDA_CHECK_LAST_ERROR(fit_poisson_gauss_kernel);
}

template <typename Device>
void def_fit_poisson_gauss(nb::module_& m) {
    m.def(
        "fit_poisson_gauss",
        [](gpu_array_c<const float, Device> values,
           gpu_array_c<const float, Device> weights,
           gpu_array_c<const float, Device> log_gamma,
           gpu_array_c<const float, Device> inits,
           gpu_array_c<float, Device> params, gpu_array_c<float, Device> loss,
           int n_guides, int n_iter, float tol, std::uintptr_t stream) {
            const int ld = static_cast<int>(values.shape(0));
            const int n_values = static_cast<int>(values.shape(1));
            const int n_starts = static_cast<int>(inits.shape(0));
            nb_require(values.ndim() == 2 && weights.ndim() == 2 &&
                           log_gamma.ndim() == 2,
                       "values, weights and log_gamma must be 2-D");
            nb_require(weights.shape(0) == ld && weights.shape(1) == n_values &&
                           log_gamma.shape(0) == ld &&
                           log_gamma.shape(1) == n_values,
                       "values, weights and log_gamma must have equal shapes");
            nb_require(inits.ndim() == 3 && inits.shape(1) == ld &&
                           inits.shape(2) == 4,
                       "inits must have shape [starts, guides, 4]");
            nb_require(params.ndim() == 3 && params.shape(0) == n_starts &&
                           params.shape(1) == ld && params.shape(2) == 4,
                       "params must have the shape of inits");
            nb_require(loss.ndim() == 2 && loss.shape(0) == n_starts &&
                           loss.shape(1) == ld,
                       "loss must have shape [starts, guides]");
            nb_require(n_guides >= 0 && n_guides <= ld,
                       "n_guides must not exceed the first dimension");
            launch_fit_poisson_gauss(
                values.data(), weights.data(), log_gamma.data(), inits.data(),
                params.data(), loss.data(), n_guides, ld, n_values, n_starts,
                n_iter, tol, (cudaStream_t)stream);
        },
        "values"_a, "weights"_a, "log_gamma"_a, "inits"_a, "params"_a, "loss"_a,
        nb::kw_only(), "n_guides"_a, "n_iter"_a, "tol"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_fit_poisson_gauss<Device>(m);
}

NB_MODULE(_guide_assignment_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
