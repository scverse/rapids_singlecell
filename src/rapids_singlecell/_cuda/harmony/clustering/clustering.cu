#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "../../cublas_helpers.cuh"
#include "../../nb_types.h"

#include "../colsum/kernels_colsum.cuh"
#include "../kmeans/kernels_kmeans.cuh"
#include "../normalize/kernels_normalize.cuh"
#include "../outer/kernels_outer.cuh"
#include "../pen/kernels_pen.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "kernels_clustering.cuh"

using namespace nb::literals;

// ---------- Launch helpers ----------

// 1D grid size capped at 8 blocks/SM
static inline int grid_1d(int n, int n_sm) {
    return std::min(n_sm * 8, (n + 255) / 256);
}

// Block dim rounded up to nearest warp, capped at 256
static inline unsigned warp_aligned_bdim(unsigned n) {
    return std::min(256u, std::max(32u, (n + 31u) / 32u * 32u));
}

// ---------- CUB temp-storage query ----------

static size_t get_cub_sort_temp_bytes(int n_cells) {
    size_t bytes = 0;
    auto* du = reinterpret_cast<unsigned int*>(1);
    auto* di = reinterpret_cast<int*>(1);
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, du, du, di, di, n_cells, 0,
                                    32);
    return bytes;
}

// ---------- Column-sum dispatch ----------

enum ColsumAlgo : int {
    COLSUM_COLUMNS = 0,
    COLSUM_ATOMICS = 1,
    COLSUM_GEMM = 2
};

template <typename T>
static inline void colsum_dispatch(int algo, cublasHandle_t handle, const T* A,
                                   T* out, const T* ones_vec, int rows,
                                   int cols, int n_sm, cudaStream_t stream) {
    T one = T(1), zero = T(0);

    switch (algo) {
        case COLSUM_COLUMNS: {
            int threads = std::min(1024, std::max(32, (rows + 31) / 32 * 32));
            colsum_kernel<T><<<std::min(cols, n_sm * 8), threads, 0, stream>>>(
                A, out, rows, cols);
            break;
        }
        case COLSUM_ATOMICS: {
            cudaMemsetAsync(out, 0, cols * sizeof(T), stream);
            int col_tiles = (cols + 31) / 32;
            int target_row_tiles =
                std::max(1, n_sm * 4 / std::max(1, col_tiles));
            int rows_per_tile =
                std::max(32, (rows + target_row_tiles - 1) / target_row_tiles);
            int row_tiles = (rows + rows_per_tile - 1) / rows_per_tile;
            colsum_atomic_kernel<T>
                <<<dim3(col_tiles, row_tiles), dim3(32, 32), 0, stream>>>(
                    A, out, rows, cols, rows_per_tile);
            break;
        }
        default:  // COLSUM_GEMM
            cublas_gemv<T>(handle, CUBLAS_OP_N, cols, rows, &one, A, cols,
                           ones_vec, 1, &zero, out, 1);
            break;
    }
}

// ---------- Scatter-add to O (add/subtract block contribution) ----------

template <typename T>
static inline void scatter_add_to_O(const T* R_buf, const int* cats_in,
                                    int current_bs, int n_clusters,
                                    int n_batches, int switcher, T* O,
                                    bool use_shared, size_t shared_bytes,
                                    int n_sm, cudaStream_t stream) {
    if (use_shared) {
        int max_blocks = std::max(1, (current_bs + 63) / 64);
        int blocks = std::min(n_sm * 4, max_blocks);
        scatter_add_shared_kernel<T><<<blocks, 256, shared_bytes, stream>>>(
            R_buf, cats_in, current_bs, n_clusters, n_batches, switcher, O);
    } else {
        size_t N = (size_t)current_bs * n_clusters;
        scatter_add_kernel<T><<<(int)((N + 255) / 256), 256, 0, stream>>>(
            R_buf, cats_in, (size_t)current_bs, (size_t)n_clusters,
            (size_t)switcher, O);
    }
}

// ---------- Clustering arguments ----------

template <typename T>
struct ClusteringArgs {
    // Input/output
    const T* Z_norm;
    T* R;
    T* E;
    T* O;
    const T* Pr_b;
    const int* cats;
    const T* theta;

    // Workspace
    T* Y;
    T* Y_norm;
    T* similarities;
    int* idx_list;
    int* idx_list_alt;
    unsigned int* sort_keys;
    unsigned int* sort_keys_alt;
    uint8_t* cub_temp;
    T* R_out_buffer;
    int* cats_in;
    T* R_in_sum;
    T* R_out_sum;
    T* penalty;
    T* obj_scalar;
    const T* ones_vec;
    T* last_obj;

    // Dimensions
    int n_cells;
    int n_pcs;
    int n_clusters;
    int n_batches;
    int block_size;
    int colsum_algo;
    T sigma;
    T tol;
    int max_iter;
    unsigned int seed;
    cudaStream_t stream;
};

// ---------- Standalone objective computation ----------

template <typename T>
static T compute_objective_impl(const T* R, const T* similarities, const T* O,
                                const T* E, const T* theta, T sigma,
                                T* obj_scalar, int n_cells, int n_clusters,
                                int n_batches, cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int n_sm = prop.multiProcessorCount;

    cudaMemsetAsync(obj_scalar, 0, sizeof(T), stream);

    // K-means error: sum(R[i] * 2 * (1 - sim[i]))
    {
        size_t n = (size_t)n_cells * n_clusters;
        kmeans_err_kernel<T><<<grid_1d((int)n, n_sm), 256, 0, stream>>>(
            R, similarities, n, obj_scalar);
    }

    // Entropy: sigma * sum(x_norm * log(x_norm + eps)), row-normalized
    // internally
    entropy_kernel<T><<<n_cells, warp_aligned_bdim(n_clusters), 0, stream>>>(
        R, sigma, n_cells, n_clusters, obj_scalar);

    // Diversity: sigma * sum(theta[b] * O[b,k] * log((O[b,k]+1)/(E[b,k]+1)))
    {
        int ob_total = n_batches * n_clusters;
        int threads = (int)warp_aligned_bdim(ob_total);
        int blocks =
            std::max(1, std::min(n_sm, (ob_total + threads - 1) / threads));
        diversity_kernel<T><<<blocks, threads, 0, stream>>>(
            O, E, theta, sigma, n_batches, n_clusters, obj_scalar);
    }

    T host_obj;
    cudaMemcpyAsync(&host_obj, obj_scalar, sizeof(T), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    return host_obj;
}

// ---------- Main clustering loop ----------

template <typename T>
static void clustering_loop_impl(const ClusteringArgs<T>& a) {
    size_t cub_temp_bytes = get_cub_sort_temp_bytes(a.n_cells);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, a.stream);

    T term = T(-2) / a.sigma;
    T one = T(1), zero = T(0);
    int ob_total = a.n_batches * a.n_clusters;  // O/E element count

    // Device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int n_sm = prop.multiProcessorCount;

    // Scatter-add: prefer shared-memory variant when O fits in smem
    size_t scatter_shared_bytes =
        (size_t)a.n_batches * a.n_clusters * sizeof(T);
    bool use_scatter_shared =
        (scatter_shared_bytes <= (size_t)prop.sharedMemPerBlock);

    // Host-side convergence tracking
    std::vector<T> objectives;
    T host_obj;
    constexpr int WINDOW_SIZE = 3;

    for (int iter = 0; iter < a.max_iter; iter++) {
        // ---- Centroids: Y = R^T @ Z_norm, then L2-normalize ----
        cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, a.n_pcs, a.n_clusters,
                       a.n_cells, &one, a.Z_norm, a.n_pcs, a.R, a.n_clusters,
                       &zero, a.Y, a.n_pcs);

        l2_row_normalize_kernel<T>
            <<<a.n_clusters, warp_aligned_bdim(a.n_pcs), 0, a.stream>>>(
                a.Y, a.Y_norm, a.n_clusters, a.n_pcs);

        // ---- Similarities: Z_norm @ Y_norm^T ----
        cublas_gemm<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, a.n_clusters,
                       a.n_cells, a.n_pcs, &one, a.Y_norm, a.n_pcs, a.Z_norm,
                       a.n_pcs, &zero, a.similarities, a.n_clusters);

        // ---- Shuffle: random permutation via PCG hash + radix sort ----
        pcg_hash_kernel<<<grid_1d(a.n_cells, n_sm), 256, 0, a.stream>>>(
            a.sort_keys, a.n_cells, a.seed + iter);
        iota_kernel<<<grid_1d(a.n_cells, n_sm), 256, 0, a.stream>>>(a.idx_list,
                                                                    a.n_cells);

        cub::DeviceRadixSort::SortPairs(
            a.cub_temp, cub_temp_bytes, a.sort_keys, a.sort_keys_alt,
            a.idx_list, a.idx_list_alt, a.n_cells, 0, 32, a.stream);

        // ---- Block loop: update R in mini-batches ----
        for (int pos = 0; pos < a.n_cells; pos += a.block_size) {
            int bs = std::min(a.block_size, a.n_cells - pos);
            const int* block_idx = a.idx_list_alt + pos;

            // Gather block: R[idx] -> R_out_buffer, cats[idx] -> cats_in
            gather_rows_kernel<T>
                <<<grid_1d(bs * a.n_clusters, n_sm), 256, 0, a.stream>>>(
                    a.R, block_idx, a.R_out_buffer, bs, a.n_clusters);
            gather_int_kernel<<<grid_1d(bs, n_sm), 256, 0, a.stream>>>(
                a.cats, block_idx, a.cats_in, bs);

            // Remove old contribution: O -= scatter(R_in), E -= Pr_b @ R_in_sum
            colsum_dispatch<T>(a.colsum_algo, handle, a.R_out_buffer,
                               a.R_in_sum, a.ones_vec, bs, a.n_clusters, n_sm,
                               a.stream);
            scatter_add_to_O<T>(a.R_out_buffer, a.cats_in, bs, a.n_clusters,
                                a.n_batches, 0, a.O, use_scatter_shared,
                                scatter_shared_bytes, n_sm, a.stream);
            outer_kernel<T><<<(ob_total + 255) / 256, 256, 0, a.stream>>>(
                a.E, a.Pr_b, a.R_in_sum, (long long)a.n_batches,
                (long long)a.n_clusters, 0LL);

            // Compute penalty and fused softmax -> R_out_buffer
            penalty_kernel<T><<<(ob_total + 255) / 256, 256, 0, a.stream>>>(
                a.E, a.O, a.theta, a.penalty, a.n_batches, a.n_clusters);
            fused_pen_norm_kernel<T, int>
                <<<bs, warp_aligned_bdim(a.n_clusters), 0, a.stream>>>(
                    a.similarities, a.penalty, a.cats_in, block_idx,
                    a.R_out_buffer, term, bs, a.n_clusters);

            // Write back and add new contribution: O += scatter(R_out), E +=
            // Pr_b @ R_out_sum
            scatter_rows_kernel<T>
                <<<grid_1d(bs * a.n_clusters, n_sm), 256, 0, a.stream>>>(
                    a.R_out_buffer, block_idx, a.R, bs, a.n_clusters);
            colsum_dispatch<T>(a.colsum_algo, handle, a.R_out_buffer,
                               a.R_out_sum, a.ones_vec, bs, a.n_clusters, n_sm,
                               a.stream);
            scatter_add_to_O<T>(a.R_out_buffer, a.cats_in, bs, a.n_clusters,
                                a.n_batches, 1, a.O, use_scatter_shared,
                                scatter_shared_bytes, n_sm, a.stream);
            outer_kernel<T><<<(ob_total + 255) / 256, 256, 0, a.stream>>>(
                a.E, a.Pr_b, a.R_out_sum, (long long)a.n_batches,
                (long long)a.n_clusters, 1LL);
        }

        // ---- Objective function (reuses similarities buffer) ----
        host_obj = compute_objective_impl(
            a.R, a.similarities, a.O, a.E, a.theta, a.sigma, a.obj_scalar,
            a.n_cells, a.n_clusters, a.n_batches, a.stream);
        objectives.push_back(host_obj);

        if (static_cast<int>(objectives.size()) >= WINDOW_SIZE + 1) {
            T obj_old = T(0), obj_new = T(0);
            for (int i = 0; i < WINDOW_SIZE; i++) {
                obj_old += objectives[objectives.size() - 2 - i];
                obj_new += objectives[objectives.size() - 1 - i];
            }
            if ((obj_old - obj_new) < a.tol * std::abs(obj_old)) break;
        }
    }

    T final_obj = objectives.empty() ? T(0) : objectives.back();
    cudaMemcpyAsync(a.last_obj, &final_obj, sizeof(T), cudaMemcpyHostToDevice,
                    a.stream);
    cublasDestroy(handle);
}

// ---------- Nanobind bindings ----------

template <typename T, typename Device>
static void register_clustering_loop(nb::module_& m) {
    m.def(
        "clustering_loop",
        [](gpu_array_c<const T, Device> Z_norm, gpu_array_c<T, Device> R,
           gpu_array_c<T, Device> E, gpu_array_c<T, Device> O,
           gpu_array_c<const T, Device> Pr_b,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const T, Device> theta, gpu_array_c<T, Device> Y,
           gpu_array_c<T, Device> Y_norm, gpu_array_c<T, Device> similarities,
           gpu_array_c<int, Device> idx_list,
           gpu_array_c<int, Device> idx_list_alt,
           gpu_array_c<unsigned int, Device> sort_keys,
           gpu_array_c<unsigned int, Device> sort_keys_alt,
           gpu_array_c<uint8_t, Device> cub_temp,
           gpu_array_c<T, Device> R_out_buffer,
           gpu_array_c<int, Device> cats_in, gpu_array_c<T, Device> R_in_sum,
           gpu_array_c<T, Device> R_out_sum, gpu_array_c<T, Device> penalty_buf,
           gpu_array_c<T, Device> obj_scalar,
           gpu_array_c<const T, Device> ones_vec,
           gpu_array_c<T, Device> last_obj, int n_cells, int n_pcs,
           int n_clusters, int n_batches, int block_size, int colsum_algo,
           double sigma, double tol, int max_iter, unsigned int seed,
           std::uintptr_t stream) {
            ClusteringArgs<T> a{
                Z_norm.data(),
                R.data(),
                E.data(),
                O.data(),
                Pr_b.data(),
                cats.data(),
                theta.data(),
                Y.data(),
                Y_norm.data(),
                similarities.data(),
                idx_list.data(),
                idx_list_alt.data(),
                sort_keys.data(),
                sort_keys_alt.data(),
                cub_temp.data(),
                R_out_buffer.data(),
                cats_in.data(),
                R_in_sum.data(),
                R_out_sum.data(),
                penalty_buf.data(),
                obj_scalar.data(),
                ones_vec.data(),
                last_obj.data(),
                n_cells,
                n_pcs,
                n_clusters,
                n_batches,
                block_size,
                colsum_algo,
                static_cast<T>(sigma),
                static_cast<T>(tol),
                max_iter,
                seed,
                (cudaStream_t)stream,
            };
            clustering_loop_impl(a);
        },
        "Z_norm"_a, nb::kw_only(), "R"_a, "E"_a, "O"_a, "Pr_b"_a, "cats"_a,
        "theta"_a, "Y"_a, "Y_norm"_a, "similarities"_a, "idx_list"_a,
        "idx_list_alt"_a, "sort_keys"_a, "sort_keys_alt"_a, "cub_temp"_a,
        "R_out_buffer"_a, "cats_in"_a, "R_in_sum"_a, "R_out_sum"_a, "penalty"_a,
        "obj_scalar"_a, "ones_vec"_a, "last_obj"_a, "n_cells"_a, "n_pcs"_a,
        "n_clusters"_a, "n_batches"_a, "block_size"_a, "colsum_algo"_a,
        "sigma"_a, "tol"_a, "max_iter"_a, "seed"_a, "stream"_a = 0);
}

template <typename T, typename Device>
static void register_compute_objective(nb::module_& m) {
    m.def(
        "compute_objective",
        [](gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> similarities,
           gpu_array_c<const T, Device> O, gpu_array_c<const T, Device> E,
           gpu_array_c<const T, Device> theta, double sigma,
           gpu_array_c<T, Device> obj_scalar, int n_cells, int n_clusters,
           int n_batches, std::uintptr_t stream) {
            return compute_objective_impl<T>(
                R.data(), similarities.data(), O.data(), E.data(), theta.data(),
                static_cast<T>(sigma), obj_scalar.data(), n_cells, n_clusters,
                n_batches, (cudaStream_t)stream);
        },
        "R"_a, nb::kw_only(), "similarities"_a, "O"_a, "E"_a, "theta"_a,
        "sigma"_a, "obj_scalar"_a, "n_cells"_a, "n_clusters"_a, "n_batches"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "get_cub_sort_temp_bytes",
        [](int n_cells) { return get_cub_sort_temp_bytes(n_cells); },
        nb::kw_only(), "n_cells"_a);

    register_clustering_loop<float, Device>(m);
    register_clustering_loop<double, Device>(m);
    register_compute_objective<float, Device>(m);
    register_compute_objective<double, Device>(m);
}

NB_MODULE(_harmony_clustering_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
