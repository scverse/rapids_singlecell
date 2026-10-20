#include <cuda_runtime.h>

#include "../nb_types.h"

#include "kernels_guide_assignment.cuh"

using namespace nb::literals;

static inline void launch_fit_poisson_gauss(
    const float* values, const float* weights, const int* n_values,
    const float* draws_rate, const float* draws_mu, const float* draws_scale,
    float* params, float* loss, int n_guides, int width, int n_init_seeds,
    int n_starts, int n_iter, float tol, cudaStream_t stream) {
    if (n_guides == 0 || n_starts == 0) return;

    dim3 block(guide_assignment::BLOCK_SIZE);
    dim3 grid(n_guides, n_starts);
    guide_assignment::fit_poisson_gauss_kernel<<<grid, block, 0, stream>>>(
        values, weights, n_values, draws_rate, draws_mu, draws_scale, params,
        loss, n_guides, width, n_init_seeds, n_iter, tol);
    CUDA_CHECK_LAST_ERROR(fit_poisson_gauss_kernel);
}

static inline void launch_select_threshold(const float* params,
                                           const float* loss,
                                           const int* max_counts, float* out,
                                           int n_guides, int n_starts,
                                           cudaStream_t stream) {
    if (n_guides == 0) return;

    dim3 block(guide_assignment::BLOCK_SIZE);
    dim3 grid(n_guides);
    guide_assignment::select_threshold_kernel<<<grid, block, 0, stream>>>(
        params, loss, max_counts, out, n_guides, n_starts);
    CUDA_CHECK_LAST_ERROR(select_threshold_kernel);
}

template <typename Device>
void def_fit_poisson_gauss(nb::module_& m) {
    m.def(
        "fit_poisson_gauss",
        [](gpu_array_c<const float, Device> values,
           gpu_array_c<const float, Device> weights,
           gpu_array_c<const int, Device> n_values,
           gpu_array_c<const float, Device> draws_rate,
           gpu_array_c<const float, Device> draws_mu,
           gpu_array_c<const float, Device> draws_scale,
           gpu_array_c<float, Device> params, gpu_array_c<float, Device> loss,
           int n_iter, float tol, std::uintptr_t stream) {
            nb_require(values.ndim() == 2 && weights.ndim() == 2 &&
                           weights.shape(0) == values.shape(0) &&
                           weights.shape(1) == values.shape(1),
                       "values and weights must be 2-D of equal shape");
            const int n_guides = static_cast<int>(values.shape(0));
            const int width = static_cast<int>(values.shape(1));
            nb_require(n_values.ndim() == 1 && n_values.shape(0) == n_guides,
                       "n_values needs one entry per guide");
            nb_require(
                draws_rate.ndim() == 3 && draws_mu.ndim() == 3 &&
                    draws_scale.ndim() == 3 &&
                    draws_rate.shape(1) == guide_assignment::PRIOR_SAMPLES &&
                    draws_rate.shape(2) == n_guides &&
                    draws_mu.shape(0) == draws_rate.shape(0) &&
                    draws_mu.shape(1) == draws_rate.shape(1) &&
                    draws_mu.shape(2) == draws_rate.shape(2) &&
                    draws_scale.shape(0) == draws_rate.shape(0) &&
                    draws_scale.shape(1) == draws_rate.shape(1) &&
                    draws_scale.shape(2) == draws_rate.shape(2),
                "draws must have shape [n_init_seeds, 15, guides]");
            const int n_init_seeds = static_cast<int>(draws_rate.shape(0));
            const int n_starts = n_init_seeds + guide_assignment::N_QUANTILES;
            nb_require(params.ndim() == 3 && params.shape(0) == n_starts &&
                           params.shape(1) == n_guides && params.shape(2) == 4,
                       "params must have shape [n_init_seeds + 5, guides, 4]");
            nb_require(loss.ndim() == 2 && loss.shape(0) == n_starts &&
                           loss.shape(1) == n_guides,
                       "loss must have shape [n_init_seeds + 5, guides]");
            launch_fit_poisson_gauss(
                values.data(), weights.data(), n_values.data(),
                draws_rate.data(), draws_mu.data(), draws_scale.data(),
                params.data(), loss.data(), n_guides, width, n_init_seeds,
                n_starts, n_iter, tol, (cudaStream_t)stream);
        },
        "values"_a, "weights"_a, "n_values"_a, "draws_rate"_a, "draws_mu"_a,
        "draws_scale"_a, "params"_a, "loss"_a, nb::kw_only(), "n_iter"_a,
        "tol"_a, "stream"_a = 0);
}

template <typename Device>
void def_select_threshold(nb::module_& m) {
    m.def(
        "select_threshold",
        [](gpu_array_c<const float, Device> params,
           gpu_array_c<const float, Device> loss,
           gpu_array_c<const int, Device> max_counts,
           gpu_array_c<float, Device> out, std::uintptr_t stream) {
            nb_require(params.ndim() == 3 && params.shape(2) == 4,
                       "params must have shape [starts, guides, 4]");
            const int n_starts = static_cast<int>(params.shape(0));
            const int n_guides = static_cast<int>(params.shape(1));
            nb_require(loss.ndim() == 2 && loss.shape(0) == n_starts &&
                           loss.shape(1) == n_guides,
                       "loss must have shape [starts, guides]");
            nb_require(
                max_counts.ndim() == 1 && max_counts.shape(0) == n_guides,
                "max_counts needs one entry per guide");
            nb_require(out.ndim() == 2 && out.shape(0) == n_guides &&
                           out.shape(1) == guide_assignment::N_OUTPUTS,
                       "out must have shape [guides, 6]");
            launch_select_threshold(params.data(), loss.data(),
                                    max_counts.data(), out.data(), n_guides,
                                    n_starts, (cudaStream_t)stream);
        },
        "params"_a, "loss"_a, "max_counts"_a, "out"_a, nb::kw_only(),
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_fit_poisson_gauss<Device>(m);
    def_select_threshold<Device>(m);
}

NB_MODULE(_guide_assignment_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
