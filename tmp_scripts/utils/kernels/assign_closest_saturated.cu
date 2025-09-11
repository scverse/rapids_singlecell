extern "C" {

__global__ void assign_closest_saturated_to_unresolved(
    const float* __restrict__ spatial,              // [n_nodes, 2] all spatial positions
    const int* __restrict__ un_unsat_indices,       // [n_un_unsat] indices of unresolved unsaturated nodes
    const int* __restrict__ sat_indices,            // [n_sat] indices of saturated nodes  
    float* __restrict__ nearest_sat,                // [n_unsat] nearest saturated for each unsaturated (update NaN entries)
    const int* __restrict__ unsat_indices,          // [n_unsat] indices of all unsaturated nodes
    int n_un_unsat,
    int n_sat,
    int n_unsat
)
{
    int un_unsat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (un_unsat_idx >= n_un_unsat) return;
    
    int unsat_node_idx = un_unsat_indices[un_unsat_idx];
    
    float min_dist = 3.402823466e+38f; // FLT_MAX
    int closest_sat_idx = -1;
    
    // Get position of this unresolved unsaturated node
    float unsat_x = spatial[unsat_node_idx * 2];
    float unsat_y = spatial[unsat_node_idx * 2 + 1];
    
    // Find closest saturated node using L1 distance
    for (int sat_idx = 0; sat_idx < n_sat; sat_idx++) {
        int sat_node_idx = sat_indices[sat_idx];
        float sat_x = spatial[sat_node_idx * 2];
        float sat_y = spatial[sat_node_idx * 2 + 1];
        
        // L1 distance (Manhattan distance)
        float dist = fabsf(unsat_x - sat_x) + fabsf(unsat_y - sat_y);
        
        if (dist < min_dist) {
            min_dist = dist;
            closest_sat_idx = sat_node_idx;
        }
    }
    
    // Find the corresponding position in nearest_sat array and update it
    for (int i = 0; i < n_unsat; i++) {
        if (unsat_indices[i] == unsat_node_idx) {
            nearest_sat[i] = (float)closest_sat_idx;
            break;
        }
    }
}

} // extern "C"
