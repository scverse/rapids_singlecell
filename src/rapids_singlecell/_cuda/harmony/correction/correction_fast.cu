#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "kernels_correction_fast.cuh"

using namespace nb::literals;

// Default threshold: use cuBLAS GEMV for row-0 scatter when n_cells >= this
// value.
static constexpr int DEFAULT_GEMV_THRESHOLD = 300000;

template <typename T>
static void correction_fast_impl(
    const T* X, const T* R, const T* O, const int* cats, const int* cat_offsets,
    const int* cell_indices, T ridge_lambda, int n_cells, int n_pcs,
    int n_clusters, int n_batches,
    // workspace (per-cluster sized)
    T* Z, T* inv_mat, T* R_col, T* Phi_t_diag_R_X, T* W,
    // gmem workspace
    T* g_factor, T* g_P_row0, cudaStream_t stream) {
    int nb1 = n_batches + 1;
    int bdim = std::min(256, std::max(32, (n_batches + 31) / 32 * 32));

    // Z = X.copy()
    cudaMemcpyAsync(Z, X, (size_t)n_cells * n_pcs * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    // cuBLAS handle for GEMM/GEMV calls
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    T one = T(1), zero = T(0);

    for (int k = 0; k < n_clusters; k++) {
        // Compute inv_mat for cluster k only
        compute_inv_mats_kernel<T>
            <<<1, bdim, 0, stream>>>(O, ridge_lambda, inv_mat, g_factor,
                                     g_P_row0, n_batches, n_clusters, k);

        // R_col = R[:, k]
        gather_column_kernel<T><<<(n_cells + 255) / 256, 256, 0, stream>>>(
            R, R_col, k, n_cells, n_clusters);

        // Zero Phi_t_diag_R_X
        cudaMemsetAsync(Phi_t_diag_R_X, 0, (size_t)nb1 * n_pcs * sizeof(T),
                        stream);

        // Row 0: sum(X[i,:] * R_col[i]) for all cells
        if (n_cells < DEFAULT_GEMV_THRESHOLD) {
            dim3 block(1024);
            dim3 grid((n_pcs + 1) / 2, 8);
            scatter_add_kernel_with_bias_cat0<T><<<grid, block, 0, stream>>>(
                X, n_cells, n_pcs, Phi_t_diag_R_X, R_col);
        } else {
            // cuBLAS GEMV: Phi_t_diag_R_X[0,:] = X^T @ R_col
            if constexpr (std::is_same_v<T, float>) {
                cublasSgemv(handle, CUBLAS_OP_N, n_pcs, n_cells, &one, X, n_pcs,
                            R_col, 1, &zero, Phi_t_diag_R_X, 1);
            } else {
                cublasDgemv(handle, CUBLAS_OP_N, n_pcs, n_cells, &one, X, n_pcs,
                            R_col, 1, &zero, Phi_t_diag_R_X, 1);
            }
        }

        // Rows 1..n_batches: per-batch biased scatter-add
        {
            dim3 block(1024);
            dim3 grid(n_batches * ((n_pcs + 1) / 2));
            scatter_add_kernel_with_bias_block<T><<<grid, block, 0, stream>>>(
                X, cat_offsets, cell_indices, n_cells, n_pcs, n_batches,
                Phi_t_diag_R_X, R_col);
        }

        // W = inv_mat @ Phi_t_diag_R_X
        // Row-major: C(nb1,n_pcs) = A(nb1,nb1) @ B(nb1,n_pcs)
        // cuBLAS col-major trick: C_cm(n_pcs,nb1) = B_cm(n_pcs,nb1) @
        // A_cm(nb1,nb1)
        {
            if constexpr (std::is_same_v<T, float>) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs, nb1, nb1,
                            &one, Phi_t_diag_R_X, n_pcs, inv_mat, nb1, &zero, W,
                            n_pcs);
            } else {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs, nb1, nb1,
                            &one, Phi_t_diag_R_X, n_pcs, inv_mat, nb1, &zero, W,
                            n_pcs);
            }
        }

        // W[0, :] = 0
        cudaMemsetAsync(W, 0, n_pcs * sizeof(T), stream);

        // Z -= R_col[cell] * W[cats[cell]+1, :]
        {
            long long n = (long long)n_cells * n_pcs;
            harmony_correction_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(
                Z, W, cats, R_col, n_cells, n_pcs);
        }
    }

    cublasDestroy(handle);
}

// ---- nanobind registration ----

template <typename T>
static void register_correction_fast(nb::module_& m) {
    m.def(
        "correction_fast",
        [](cuda_array_c<const T> X, cuda_array_c<const T> R,
           cuda_array_c<const T> O, cuda_array_c<const int> cats,
           cuda_array_c<const int> cat_offsets,
           cuda_array_c<const int> cell_indices, double ridge_lambda,
           int n_cells, int n_pcs, int n_clusters, int n_batches,
           // workspace
           cuda_array_c<T> Z, cuda_array_c<T> inv_mat, cuda_array_c<T> R_col,
           cuda_array_c<T> Phi_t_diag_R_X, cuda_array_c<T> W,
           // gmem workspace
           cuda_array_c<T> g_factor, cuda_array_c<T> g_P_row0,
           std::uintptr_t stream) {
            correction_fast_impl<T>(
                X.data(), R.data(), O.data(), cats.data(), cat_offsets.data(),
                cell_indices.data(), static_cast<T>(ridge_lambda), n_cells,
                n_pcs, n_clusters, n_batches, Z.data(), inv_mat.data(),
                R_col.data(), Phi_t_diag_R_X.data(), W.data(), g_factor.data(),
                g_P_row0.data(), (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "cats"_a, "cat_offsets"_a,
        "cell_indices"_a, "ridge_lambda"_a, "n_cells"_a, "n_pcs"_a,
        "n_clusters"_a, "n_batches"_a, "Z"_a, "inv_mat"_a, "R_col"_a,
        "Phi_t_diag_R_X"_a, "W"_a, "g_factor"_a, "g_P_row0"_a, "stream"_a = 0);

    // Expose single-cluster inv_mat computation for testing
    m.def(
        "compute_inv_mat",
        [](cuda_array_c<const T> O, double ridge_lambda, int n_batches,
           int n_clusters, int cluster_k, cuda_array_c<T> inv_mat,
           cuda_array_c<T> g_factor, cuda_array_c<T> g_P_row0,
           std::uintptr_t stream) {
            int bdim = std::min(256, std::max(32, (n_batches + 31) / 32 * 32));
            compute_inv_mats_kernel<T><<<1, bdim, 0, (cudaStream_t)stream>>>(
                O.data(), static_cast<T>(ridge_lambda), inv_mat.data(),
                g_factor.data(), g_P_row0.data(), n_batches, n_clusters,
                cluster_k);
        },
        "O"_a, nb::kw_only(), "ridge_lambda"_a, "n_batches"_a, "n_clusters"_a,
        "cluster_k"_a, "inv_mat"_a, "g_factor"_a, "g_P_row0"_a, "stream"_a = 0);
}

NB_MODULE(_harmony_correction_cuda, m) {
    register_correction_fast<float>(m);
    register_correction_fast<double>(m);
}
