#include <cuda_runtime.h>
#include "../nb_types.h"

using namespace nb::literals;

__global__ void auc_kernel(const int* __restrict__ ranks, int R, int C,
                           const int* __restrict__ cnct,
                           const int* __restrict__ starts,
                           const int* __restrict__ lens, int n_sets, int n_up,
                           const float* __restrict__ max_aucs,
                           float* __restrict__ es) {
    const int set = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (set >= n_sets || row >= R) return;

    const int start = starts[set];
    const int end = start + lens[set];

    int r = 0;
    int s = 0;

    for (int i = start; i < end; ++i) {
        const int g = cnct[i];
        const int rk = ranks[row * C + g];
        if (rk <= n_up) {
            r += 1;
            s += rk;
        }
    }
    const float val =
        (float)((static_cast<long long>(r) * n_up) - s) / max_aucs[set];
    es[row * n_sets + set] = val;
}

static inline void launch_auc(const int* ranks, int R, int C, const int* cnct,
                              const int* starts, const int* lens, int n_sets,
                              int n_up, const float* max_aucs, float* es,
                              cudaStream_t stream) {
    dim3 block(32);
    dim3 grid((unsigned)n_sets, (unsigned)((R + block.x - 1) / block.x));
    auc_kernel<<<grid, block, 0, stream>>>(ranks, R, C, cnct, starts, lens,
                                           n_sets, n_up, max_aucs, es);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "auc",
        [](gpu_array_c<const int, Device> ranks, int R, int C,
           gpu_array_c<const int, Device> cnct,
           gpu_array_c<const int, Device> starts,
           gpu_array_c<const int, Device> lens, int n_sets, int n_up,
           gpu_array_c<const float, Device> max_aucs,
           gpu_array_c<float, Device> es, std::uintptr_t stream) {
            launch_auc(ranks.data(), R, C, cnct.data(), starts.data(),
                       lens.data(), n_sets, n_up, max_aucs.data(), es.data(),
                       (cudaStream_t)stream);
        },
        "ranks"_a, nb::kw_only(), "R"_a, "C"_a, "cnct"_a, "starts"_a, "lens"_a,
        "n_sets"_a, "n_up"_a, "max_aucs"_a, "es"_a, "stream"_a = 0);
}

NB_MODULE(_aucell_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
