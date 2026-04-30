#include <cuda_runtime.h>
#include <math_constants.h>

#include <limits>
#include <stdexcept>
#include <string>

#include "../nb_types.h"

#include "kernels_gmm.cuh"

using namespace nb::literals;

constexpr int TILE = 16;
constexpr int E_STEP_BLOCK = 64;
constexpr int NORMALIZE_BLOCK = 32;
constexpr int M_STEP_CHUNK_ROWS = 1024;
constexpr int M_STEP_CHUNKED_MIN_ROWS = 32768;

static inline void cuda_check_runtime(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) +
                                 " failed: " + cudaGetErrorString(err));
    }
}

template <typename T>
static inline void launch_e_step(const T* X, const T* weights, const T* means,
                                 const T* prec_chol, const T* log_det_half,
                                 int n, int d, int K, T* log_prob, T* resp,
                                 T* ll_per_cell, cudaStream_t stream) {
    if (n == 0 || d == 0 || K == 0) return;
    if (d > 64) {
        throw std::runtime_error(
            "_gmm_cuda.e_step supports at most 64 features");
    }
    {
        size_t shmem = (size_t)(d + d * d) * sizeof(T);
        dim3 block(E_STEP_BLOCK);
        dim3 grid(K, (n + E_STEP_BLOCK - 1) / E_STEP_BLOCK);
        e_step_log_prob_kernel<T><<<grid, block, shmem, stream>>>(
            X, weights, means, prec_chol, log_det_half, n, d, K, log_prob);
        CUDA_CHECK_LAST_ERROR(e_step_log_prob_kernel);
    }
    {
        dim3 block(NORMALIZE_BLOCK);
        dim3 grid(n);
        e_step_normalize_kernel<T>
            <<<grid, block, 0, stream>>>(log_prob, n, K, resp, ll_per_cell);
        CUDA_CHECK_LAST_ERROR(e_step_normalize_kernel);
    }
}

template <typename T>
static inline void launch_m_step(const T* resp, const T* X, int n, int d, int K,
                                 T reg_covar, T* weights, T* means,
                                 T* covariances, T* workspace_N_k,
                                 T* workspace_num, cudaStream_t stream) {
    if (n == 0 || d == 0 || K == 0) return;
    int tiles_d = (d + TILE - 1) / TILE;
    T eps = std::numeric_limits<T>::epsilon();
    if (n < M_STEP_CHUNKED_MIN_ROWS) {
        dim3 block(TILE, TILE);
        dim3 grid(K, tiles_d, tiles_d);
        m_step_fused_kernel<T, TILE><<<grid, block, 0, stream>>>(
            resp, X, n, d, K, workspace_N_k, workspace_num, covariances);
        CUDA_CHECK_LAST_ERROR(m_step_fused_kernel);
    } else {
        size_t n_k_bytes = static_cast<size_t>(K) * sizeof(T);
        size_t num_bytes = static_cast<size_t>(K) * d * sizeof(T);
        size_t cov_bytes = static_cast<size_t>(K) * d * d * sizeof(T);
        cuda_check_runtime(cudaMemsetAsync(workspace_N_k, 0, n_k_bytes, stream),
                           "cudaMemsetAsync(workspace_N_k)");
        cuda_check_runtime(cudaMemsetAsync(workspace_num, 0, num_bytes, stream),
                           "cudaMemsetAsync(workspace_num)");
        cuda_check_runtime(cudaMemsetAsync(covariances, 0, cov_bytes, stream),
                           "cudaMemsetAsync(covariances)");
        int chunks = (n + M_STEP_CHUNK_ROWS - 1) / M_STEP_CHUNK_ROWS;
        dim3 block(TILE, TILE);
        dim3 grid(K, tiles_d * tiles_d, chunks);
        m_step_chunked_atomic_kernel<T, TILE><<<grid, block, 0, stream>>>(
            resp, X, n, d, K, tiles_d, M_STEP_CHUNK_ROWS, workspace_N_k,
            workspace_num, covariances);
        CUDA_CHECK_LAST_ERROR(m_step_chunked_atomic_kernel);
    }
    {
        int threads = 256;
        dim3 block(threads);
        dim3 grid(K);
        m_step_finalize_kernel<T><<<grid, block, 0, stream>>>(
            workspace_N_k, workspace_num, covariances, weights, means,
            reg_covar, eps, n, d, K);
        CUDA_CHECK_LAST_ERROR(m_step_finalize_kernel);
    }
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "e_step",
        [](gpu_array_c<const float, Device> X,
           gpu_array_c<const float, Device> weights,
           gpu_array_c<const float, Device> means,
           gpu_array_c<const float, Device> prec_chol,
           gpu_array_c<const float, Device> log_det_half,
           gpu_array_c<float, Device> log_prob, gpu_array_c<float, Device> resp,
           gpu_array_c<float, Device> ll_per_cell, int n, int d, int K,
           std::uintptr_t stream) {
            launch_e_step<float>(X.data(), weights.data(), means.data(),
                                 prec_chol.data(), log_det_half.data(), n, d, K,
                                 log_prob.data(), resp.data(),
                                 ll_per_cell.data(), (cudaStream_t)stream);
        },
        "X"_a, "weights"_a, "means"_a, "prec_chol"_a, "log_det_half"_a,
        "log_prob"_a, "resp"_a, "ll_per_cell"_a, nb::kw_only(), "n"_a, "d"_a,
        "K"_a, "stream"_a = 0);

    m.def(
        "e_step",
        [](gpu_array_c<const double, Device> X,
           gpu_array_c<const double, Device> weights,
           gpu_array_c<const double, Device> means,
           gpu_array_c<const double, Device> prec_chol,
           gpu_array_c<const double, Device> log_det_half,
           gpu_array_c<double, Device> log_prob,
           gpu_array_c<double, Device> resp,
           gpu_array_c<double, Device> ll_per_cell, int n, int d, int K,
           std::uintptr_t stream) {
            launch_e_step<double>(X.data(), weights.data(), means.data(),
                                  prec_chol.data(), log_det_half.data(), n, d,
                                  K, log_prob.data(), resp.data(),
                                  ll_per_cell.data(), (cudaStream_t)stream);
        },
        "X"_a, "weights"_a, "means"_a, "prec_chol"_a, "log_det_half"_a,
        "log_prob"_a, "resp"_a, "ll_per_cell"_a, nb::kw_only(), "n"_a, "d"_a,
        "K"_a, "stream"_a = 0);

    m.def(
        "m_step",
        [](gpu_array_c<const float, Device> resp,
           gpu_array_c<const float, Device> X,
           gpu_array_c<float, Device> weights, gpu_array_c<float, Device> means,
           gpu_array_c<float, Device> covariances,
           gpu_array_c<float, Device> N_k_workspace,
           gpu_array_c<float, Device> num_workspace, int n, int d, int K,
           float reg_covar, std::uintptr_t stream) {
            launch_m_step<float>(resp.data(), X.data(), n, d, K, reg_covar,
                                 weights.data(), means.data(),
                                 covariances.data(), N_k_workspace.data(),
                                 num_workspace.data(), (cudaStream_t)stream);
        },
        "resp"_a, "X"_a, "weights"_a, "means"_a, "covariances"_a,
        "N_k_workspace"_a, "num_workspace"_a, nb::kw_only(), "n"_a, "d"_a,
        "K"_a, "reg_covar"_a, "stream"_a = 0);

    m.def(
        "m_step",
        [](gpu_array_c<const double, Device> resp,
           gpu_array_c<const double, Device> X,
           gpu_array_c<double, Device> weights,
           gpu_array_c<double, Device> means,
           gpu_array_c<double, Device> covariances,
           gpu_array_c<double, Device> N_k_workspace,
           gpu_array_c<double, Device> num_workspace, int n, int d, int K,
           double reg_covar, std::uintptr_t stream) {
            launch_m_step<double>(resp.data(), X.data(), n, d, K, reg_covar,
                                  weights.data(), means.data(),
                                  covariances.data(), N_k_workspace.data(),
                                  num_workspace.data(), (cudaStream_t)stream);
        },
        "resp"_a, "X"_a, "weights"_a, "means"_a, "covariances"_a,
        "N_k_workspace"_a, "num_workspace"_a, nb::kw_only(), "n"_a, "d"_a,
        "K"_a, "reg_covar"_a, "stream"_a = 0);
}

NB_MODULE(_gmm_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
