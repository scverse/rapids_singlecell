// kernels/edistance_kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// Find closest saturated node for each unsaturated node using L1 distance
__global__ void find_closest_saturated_nodes(
    const float* __restrict__ spatial_unsat,     // [n_unsat, 2] unsaturated node positions
    const float* __restrict__ spatial_sat,       // [n_sat, 2] saturated node positions  
    const int* __restrict__ unsat_indices,       // [n_unsat] unsaturated node indices
    const int* __restrict__ sat_indices,         // [n_sat] saturated node indices
    int* __restrict__ closest_sat,               // [n_unsat] output: closest saturated node for each unsat
    int n_unsat,
    int n_sat
) 
{
    int unsat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (unsat_idx >= n_unsat) return;
    
    float min_dist = 3.402823466e+38f;
    int closest_idx = -1;
    
    // Get position of this unsaturated node
    float unsat_x = spatial_unsat[unsat_idx * 2];
    float unsat_y = spatial_unsat[unsat_idx * 2 + 1];
    
    // Find closest saturated node using L1 distance
    for (int sat_idx = 0; sat_idx < n_sat; sat_idx++) {
        float sat_x = spatial_sat[sat_idx * 2];
        float sat_y = spatial_sat[sat_idx * 2 + 1];
        
        // L1 distance (Manhattan distance)
        float dist = fabsf(unsat_x - sat_x) + fabsf(unsat_y - sat_y);
        
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = sat_indices[sat_idx];
        }
    }
    
    closest_sat[unsat_idx] = closest_idx;
}

} // extern "C"