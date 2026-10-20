#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace guide_assignment {

constexpr int BLOCK_SIZE = 128;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;
constexpr int NEWTON_STEPS = 12;
constexpr float LOG_SQRT_2PI = 0.91893853320467274f;
constexpr float SCALE_FLOOR = 1e-3f;
constexpr float LOG_SCALE_FLOOR = -6.9077552789821368f;
constexpr float WEIGHT_FLOOR = 1e-6f;

struct Params {
    float rate;
    float mu;
    float scale;
    float w_norm;
};

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Block-wide sums of three values, identical on every thread afterwards.
__device__ inline void block_sum3(float (&s)[3], float (*partials)[3]) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        s[k] = warp_sum(s[k]);
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            partials[warp][k] = s[k];
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            total += partials[w][k];
        }
        s[k] = total;
    }
}

// Log mixture density of one log2 count and the Gaussian responsibility.
__device__ inline void mixture_terms(float value, float log_gamma,
                                     const Params& p, float log_rate,
                                     float log_scale, float log_w,
                                     float log1m_w, float& log_mix,
                                     float& resp) {
    const float log_pois = value * log_rate - p.rate - log_gamma;
    const float z = (value - p.mu) / p.scale;
    const float log_norm = -LOG_SQRT_2PI - log_scale - 0.5f * z * z;
    const float log_norm_weighted = log_w + log_norm;
    const float log_pois_weighted = log1m_w + log_pois;
    const float upper = fmaxf(log_norm_weighted, log_pois_weighted);
    log_mix =
        upper + log1pf(expf(-fabsf(log_norm_weighted - log_pois_weighted)));
    resp = expf(log_norm_weighted - log_mix);
}

// Log prior density without its additive constants.
__device__ inline float log_prior(const Params& p) {
    const float log_scale = logf(p.scale);
    const float log_rate = logf(p.rate);
    const float d = (p.mu - 3.0f) / 2.0f;
    return -0.1f * log1pf(-p.w_norm) - 0.9f * logf(p.w_norm) - 0.5f * d * d -
           log_scale - 0.5f * (log_scale - 2.0f) * (log_scale - 2.0f) -
           log_rate - 0.5f * log_rate * log_rate;
}

// MAP log scale of the Gaussian: the root of a convex decreasing function.
__device__ inline float solve_log_scale(float sum_squares, float n_norm) {
    float v = 0.5f * logf(fmaxf(sum_squares, 1e-12f) / fmaxf(n_norm, 1e-12f));
    v = fminf(fmaxf(v, LOG_SCALE_FLOOR), 10.0f);
    for (int i = 0; i < NEWTON_STEPS; ++i) {
        const float e = sum_squares * expf(-2.0f * v);
        const float f = e - (v - 2.0f) - n_norm - 1.0f;
        v = fminf(fmaxf(v - f / (-2.0f * e - 1.0f), LOG_SCALE_FLOOR), 10.0f);
    }
    return v;
}

// MAP log rate of the Poisson: the root of a concave decreasing function.
__device__ inline float solve_log_rate(float sum_values, float n_pois) {
    float log_rate = logf(fmaxf(sum_values, 1e-12f) / fmaxf(n_pois, 1e-12f));
    log_rate = fminf(fmaxf(log_rate, -20.0f), 10.0f);
    for (int i = 0; i < NEWTON_STEPS; ++i) {
        const float e = n_pois * expf(log_rate);
        const float g = sum_values - 1.0f - log_rate - e;
        log_rate = fminf(fmaxf(log_rate - g / (-1.0f - e), -20.0f), 10.0f);
    }
    return log_rate;
}

// Sums of w * log_mix, w * resp and w * resp * value, or of w * resp * (value -
// center)^2.
template <bool SECOND_PASS>
__device__ inline void e_step_sums(const float* __restrict__ values,
                                   const float* __restrict__ weights,
                                   const float* __restrict__ log_gamma,
                                   int n_values, const Params& p, float center,
                                   float (&s)[3], float (*partials)[3]) {
    const float log_rate = logf(p.rate);
    const float log_scale = logf(p.scale);
    const float log_w = logf(p.w_norm);
    const float log1m_w = log1pf(-p.w_norm);
    s[0] = 0.0f;
    s[1] = 0.0f;
    s[2] = 0.0f;
    for (int j = threadIdx.x; j < n_values; j += BLOCK_SIZE) {
        float log_mix;
        float resp;
        mixture_terms(values[j], log_gamma[j], p, log_rate, log_scale, log_w,
                      log1m_w, log_mix, resp);
        const float w_resp = weights[j] * resp;
        if (SECOND_PASS) {
            const float d = values[j] - center;
            s[0] += w_resp * d * d;
        } else {
            s[0] += weights[j] * log_mix;
            s[1] += w_resp;
            s[2] += w_resp * values[j];
        }
    }
    block_sum3(s, partials);
}

// MAP-EM of one (start, guide) until the relative loss change is below tol;
// layouts [starts, ld, 4] and [starts, ld].
__device__ inline void fit_poisson_gauss_block(
    const float* __restrict__ values, const float* __restrict__ weights,
    const float* __restrict__ log_gamma, const float* __restrict__ inits,
    float* __restrict__ params_out, float* __restrict__ loss_out, int ld,
    int n_values, int n_iter, float tol) {
    const int guide = blockIdx.x;
    const int start = blockIdx.y;
    const long long offset = static_cast<long long>(guide) * n_values;
    const float* v = values + offset;
    const float* w = weights + offset;
    const float* lg = log_gamma + offset;
    const long long slot = static_cast<long long>(start) * ld + guide;

    __shared__ float partials[NUM_WARPS][3];

    Params p{inits[slot * 4], inits[slot * 4 + 1], inits[slot * 4 + 2],
             inits[slot * 4 + 3]};

    float s[3] = {0.0f, 0.0f, 0.0f};
    for (int j = threadIdx.x; j < n_values; j += BLOCK_SIZE) {
        s[0] += w[j];
        s[1] += w[j] * v[j];
    }
    block_sum3(s, partials);
    const float sum_w = s[0];
    const float sum_wv = s[1];

    float previous = INFINITY;
    float current = 1e30f;
    for (int iteration = 0; iteration < n_iter; ++iteration) {
        e_step_sums<false>(v, w, lg, n_values, p, 0.0f, s, partials);
        const float loss = -(s[0] + log_prior(p));
        const float n_norm = s[1];
        const float sum_resp_values = s[2];
        const float n_pois = sum_w - n_norm;
        const float w_new = fminf(
            fmaxf((n_norm - 0.9f) / (n_norm + n_pois - 1.0f), WEIGHT_FLOOR),
            1.0f - WEIGHT_FLOOR);
        const float precision = 1.0f / (p.scale * p.scale);
        const float mu_new = (sum_resp_values * precision + 0.75f) /
                             (n_norm * precision + 0.25f);
        e_step_sums<true>(v, w, lg, n_values, p, mu_new, s, partials);
        const float scale_new =
            fmaxf(expf(solve_log_scale(s[0], n_norm)), SCALE_FLOOR);
        const float rate_new =
            expf(solve_log_rate(sum_wv - sum_resp_values, n_pois));
        p = Params{rate_new, mu_new, scale_new, w_new};
        previous = current;
        current = loss;
        if (!(fabsf(previous - current) > tol * fabsf(current))) {
            break;
        }
    }

    e_step_sums<false>(v, w, lg, n_values, p, 0.0f, s, partials);
    if (threadIdx.x == 0) {
        params_out[slot * 4] = p.rate;
        params_out[slot * 4 + 1] = p.mu;
        params_out[slot * 4 + 2] = p.scale;
        params_out[slot * 4 + 3] = p.w_norm;
        loss_out[slot] = -(s[0] + log_prior(p));
    }
}

__global__ void fit_poisson_gauss_kernel(const float* __restrict__ values,
                                         const float* __restrict__ weights,
                                         const float* __restrict__ log_gamma,
                                         const float* __restrict__ inits,
                                         float* __restrict__ params_out,
                                         float* __restrict__ loss_out, int ld,
                                         int n_values, int n_iter, float tol) {
    fit_poisson_gauss_block(values, weights, log_gamma, inits, params_out,
                            loss_out, ld, n_values, n_iter, tol);
}

}  // namespace guide_assignment
