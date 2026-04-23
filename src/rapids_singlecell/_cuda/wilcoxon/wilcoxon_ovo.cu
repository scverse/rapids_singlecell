#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_common.cuh"
#include "kernels_wilcoxon.cuh"
#include "kernels_wilcoxon_ovo.cuh"

using namespace nb::literals;

#include "wilcoxon_ovo_kernels.cuh"
#include "wilcoxon_ovo_device_dense.cuh"
#include "wilcoxon_ovo_device_sparse.cuh"
#include "wilcoxon_ovo_host_sparse.cuh"
#include "wilcoxon_ovo_host_dense.cuh"
#include "wilcoxon_ovo_bindings.cuh"

NB_MODULE(_wilcoxon_ovo_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
