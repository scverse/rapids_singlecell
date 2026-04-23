#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_common.cuh"
#include "kernels_wilcoxon.cuh"

using namespace nb::literals;

#include "wilcoxon_ovr_kernels.cuh"
#include "wilcoxon_ovr_dense.cuh"
#include "wilcoxon_ovr_sparse.cuh"
#include "wilcoxon_ovr_bindings.cuh"

NB_MODULE(_wilcoxon_ovr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
