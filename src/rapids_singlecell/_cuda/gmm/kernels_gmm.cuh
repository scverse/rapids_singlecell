#pragma once

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Per-(n, k) E-step log-probability.
//
// Each block (k, n_chunk) caches means[k] and prec_chol[k] in shared memory,
// then each thread computes mahalanobis for one cell against the cached
// component. Output is row-major log_prob[n, k] with the log-weight already
// folded in:
//
//   y[j]        = Σ_d (X[n, d] − means[k, d]) · prec_chol[k, d, j]
//   mahal[n, k] = Σ_j y[j]²
//   log_prob[n, k] = −0.5·d·log(2π) + log_det_half[k] − 0.5·mahal +
//   log(weights[k])
//
// A separate normalize kernel does the per-row logsumexp.
// ----------------------------------------------------------------------------

constexpr float LOG_2PI_F = 1.8378770664093453f;
constexpr double LOG_2PI_D = 1.8378770664093453;

template <typename T>
__device__ __forceinline__ T log_2pi_const();
template <>
__device__ __forceinline__ float log_2pi_const<float>() {
    return LOG_2PI_F;
}
template <>
__device__ __forceinline__ double log_2pi_const<double>() {
    return LOG_2PI_D;
}

template <typename T>
__global__ void e_step_log_prob_kernel(
    const T* __restrict__ X,             // (n, d) row-major
    const T* __restrict__ weights,       // (K,)
    const T* __restrict__ means,         // (K, d)
    const T* __restrict__ prec_chol,     // (K, d, d) row-major; cov_inv =
                                         // chol·cholᵀ
    const T* __restrict__ log_det_half,  // (K,)
    int n, int d, int K,
    T* __restrict__ log_prob  // (n, K)
) {
    int k = blockIdx.x;
    int n_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ unsigned char smem_raw[];
    T* sh_mean = reinterpret_cast<T*>(smem_raw);
    T* sh_pc = sh_mean + d;

    // Cooperatively load means[k] and prec_chol[k] into shared memory.
    for (int i = tid; i < d; i += blockDim.x) sh_mean[i] = means[k * d + i];
    int pc_size = d * d;
    for (int i = tid; i < pc_size; i += blockDim.x)
        sh_pc[i] = prec_chol[k * pc_size + i];

    T log_w = log(weights[k]);
    T ldh = log_det_half[k];
    T const_term = T(-0.5) * T(d) * log_2pi_const<T>() + ldh + log_w;

    __syncthreads();

    if (n_idx >= n) return;

    // Compute mahal = || (X[n] - μ_k) · prec_chol[k] ||²
    T centered_vals[64];
    for (int dd = 0; dd < d; ++dd)
        centered_vals[dd] = X[n_idx * d + dd] - sh_mean[dd];

    T mahal = T(0);
    for (int j = 0; j < d; ++j) {
        T y = T(0);
        for (int dd = 0; dd < d; ++dd) {
            y += centered_vals[dd] * sh_pc[dd * d + j];
        }
        mahal += y * y;
    }
    log_prob[n_idx * K + k] = const_term - T(0.5) * mahal;
}

// ----------------------------------------------------------------------------
// Per-cell logsumexp normalize: resp[n, k] = exp(log_prob[n, k] − logΣ_k).
// Also writes per-cell log-likelihood (= logΣ_k) into ll_per_cell for later
// reduction. One block per cell; threads stride across K.
// ----------------------------------------------------------------------------

template <typename T>
__global__ void e_step_normalize_kernel(
    const T* __restrict__ log_prob,  // (n, K)
    int n, int K,
    T* __restrict__ resp,        // (n, K)
    T* __restrict__ ll_per_cell  // (n,)
) {
    int n_idx = blockIdx.x;
    if (n_idx >= n) return;
    int tid = threadIdx.x;

    __shared__ T sh_max;
    __shared__ T sh_sum;

    // pass 1: max over K
    T local_max = -CUDART_INF_F;
    for (int k = tid; k < K; k += blockDim.x) {
        T v = log_prob[n_idx * K + k];
        if (v > local_max) local_max = v;
    }
    // warp + block reduce max
    for (int off = 16; off > 0; off >>= 1) {
        T other = __shfl_down_sync(0xffffffff, local_max, off);
        if (other > local_max) local_max = other;
    }
    if (tid == 0) sh_max = local_max;
    __syncthreads();
    T mx = sh_max;

    // pass 2: sum exp(log_prob - max)
    T local_sum = T(0);
    for (int k = tid; k < K; k += blockDim.x) {
        local_sum += exp(log_prob[n_idx * K + k] - mx);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if (tid == 0) {
        sh_sum = local_sum;
        T log_total = log(local_sum) + mx;
        ll_per_cell[n_idx] = log_total;
    }
    __syncthreads();
    T log_total = log(sh_sum) + mx;

    // pass 3: write normalized responsibilities
    for (int k = tid; k < K; k += blockDim.x) {
        resp[n_idx * K + k] = exp(log_prob[n_idx * K + k] - log_total);
    }
}

// ----------------------------------------------------------------------------
// Direct M-step kernel for smaller inputs.
// ----------------------------------------------------------------------------

template <typename T, int TILE>
__global__ void m_step_fused_kernel(const T* __restrict__ resp,
                                    const T* __restrict__ X, int n, int d,
                                    int K, T* __restrict__ N_k,
                                    T* __restrict__ num, T* __restrict__ sm) {
    int k = blockIdx.x;
    int i_tile = blockIdx.y * TILE;
    int j_tile = blockIdx.z * TILE;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    int i = i_tile + ti;
    int j = j_tile + tj;
    bool valid = (i < d) && (j < d);
    bool emit_cov = j_tile >= i_tile;

    bool emit_num = (j_tile == 0);
    bool emit_Nk = (i_tile == 0) && (j_tile == 0);

    if (!emit_cov && !emit_num) return;

    __shared__ T Xi[TILE][TILE];
    __shared__ T Xj[TILE][TILE];
    __shared__ T r[TILE];

    T accum_sm = T(0);
    T accum_num = T(0);
    T block_Nk = T(0);

    for (int n_base = 0; n_base < n; n_base += TILE) {
        int n_in = min(TILE, n - n_base);

        T xi_v = T(0), xj_v = T(0);
        if (ti < n_in && (i_tile + tj) < d)
            xi_v = X[(n_base + ti) * d + i_tile + tj];
        if (emit_cov && ti < n_in && (j_tile + tj) < d)
            xj_v = X[(n_base + ti) * d + j_tile + tj];
        Xi[ti][tj] = xi_v;
        Xj[ti][tj] = xj_v;

        if (ti == 0) {
            r[tj] = (tj < n_in) ? resp[(n_base + tj) * K + k] : T(0);
        }
        __syncthreads();

        if (valid) {
#pragma unroll
            for (int t = 0; t < TILE; ++t) {
                T rt = r[t];
                T xi_t = Xi[t][ti];
                if (emit_cov) {
                    T xj_t = Xj[t][tj];
                    accum_sm += rt * xi_t * xj_t;
                }
                if (emit_num && tj == 0) accum_num += rt * xi_t;
            }
        }

        if (emit_Nk && ti == 0 && tj == 0) {
            T chunk_sum = T(0);
#pragma unroll
            for (int t = 0; t < TILE; ++t) chunk_sum += r[t];
            block_Nk += chunk_sum;
        }
        __syncthreads();
    }

    if (emit_cov && valid) sm[k * d * d + i * d + j] = accum_sm;
    if (emit_num && tj == 0 && i < d) num[k * d + i] = accum_num;
    if (emit_Nk && ti == 0 && tj == 0) N_k[k] = block_Nk;
}

template <typename T, int TILE>
__global__ void m_step_chunked_atomic_kernel(
    const T* __restrict__ resp, const T* __restrict__ X, int n, int d, int K,
    int tiles_d, int chunk_size, T* __restrict__ N_k, T* __restrict__ num,
    T* __restrict__ sm) {
    int k = blockIdx.x;
    int tile_pair = blockIdx.y;
    int chunk = blockIdx.z;
    int i_tile = (tile_pair / tiles_d) * TILE;
    int j_tile = (tile_pair % tiles_d) * TILE;
    int chunk_start = chunk * chunk_size;
    int chunk_end = min(n, chunk_start + chunk_size);

    int ti = threadIdx.y;
    int tj = threadIdx.x;
    int i = i_tile + ti;
    int j = j_tile + tj;
    bool valid = (i < d) && (j < d);
    bool emit_cov = j_tile >= i_tile;

    bool emit_num = (j_tile == 0);
    bool emit_Nk = (i_tile == 0) && (j_tile == 0);

    if (!emit_cov && !emit_num) return;

    __shared__ T Xi[TILE][TILE];
    __shared__ T Xj[TILE][TILE];
    __shared__ T r[TILE];

    T accum_sm = T(0);
    T accum_num = T(0);
    T block_Nk = T(0);

    for (int n_base = chunk_start; n_base < chunk_end; n_base += TILE) {
        int n_in = min(TILE, chunk_end - n_base);

        T xi_v = T(0), xj_v = T(0);
        if (ti < n_in && (i_tile + tj) < d)
            xi_v = X[(n_base + ti) * d + i_tile + tj];
        if (emit_cov && ti < n_in && (j_tile + tj) < d)
            xj_v = X[(n_base + ti) * d + j_tile + tj];
        Xi[ti][tj] = xi_v;
        Xj[ti][tj] = xj_v;

        if (ti == 0) {
            r[tj] = (tj < n_in) ? resp[(n_base + tj) * K + k] : T(0);
        }
        __syncthreads();

        if (valid) {
#pragma unroll
            for (int t = 0; t < TILE; ++t) {
                T rt = r[t];
                T xi_t = Xi[t][ti];
                if (emit_cov) {
                    T xj_t = Xj[t][tj];
                    accum_sm += rt * xi_t * xj_t;
                }
                if (emit_num && tj == 0) accum_num += rt * xi_t;
            }
        }

        if (emit_Nk && ti == 0 && tj == 0) {
            T chunk_sum = T(0);
#pragma unroll
            for (int t = 0; t < TILE; ++t) chunk_sum += r[t];
            block_Nk += chunk_sum;
        }
        __syncthreads();
    }

    if (emit_cov && valid) atomicAdd(&sm[k * d * d + i * d + j], accum_sm);
    if (emit_num && tj == 0 && i < d) atomicAdd(&num[k * d + i], accum_num);
    if (emit_Nk && ti == 0 && tj == 0) atomicAdd(&N_k[k], block_Nk);
}

template <typename T>
__global__ void m_step_finalize_kernel(const T* __restrict__ N_k,
                                       const T* __restrict__ num,
                                       T* __restrict__ sm_to_cov,
                                       T* __restrict__ weights,
                                       T* __restrict__ means, T reg_covar,
                                       T eps, int n, int d, int K) {
    int k = blockIdx.x;
    int tid = threadIdx.x;

    T Nk = N_k[k] + T(10) * eps;
    T inv_Nk = T(1) / Nk;

    if (tid == 0) weights[k] = Nk / T(n);

    for (int i = tid; i < d; i += blockDim.x)
        means[k * d + i] = num[k * d + i] * inv_Nk;
    __syncthreads();

    int total = d * d;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int i = idx / d;
        int j = idx % d;
        if (i > j) continue;
        T mi = means[k * d + i];
        T mj = means[k * d + j];
        T v = sm_to_cov[k * d * d + idx] * inv_Nk - mi * mj;
        if (i == j) v += reg_covar;
        sm_to_cov[k * d * d + idx] = v;
        if (i != j) sm_to_cov[k * d * d + j * d + i] = v;
    }
}
