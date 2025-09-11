extern "C" {

    __global__ void compute_closest_saturated_distances(
        const float* __restrict__ spatial_un_unsat,    // [n_un_unsat, 2] positions of unresolved unsaturated nodes
        const float* __restrict__ spatial_sat,         // [n_sat, 2] positions of saturated nodes
        const int* __restrict__ sat_indices,           // [n_sat] indices of saturated nodes
        float* __restrict__ nearest_sat_indices,       // [n_unsat] output: update NaN entries with closest sat indices
        const int* __restrict__ un_unsat_mask,         // [n_unsat] mask indicating which entries are NaN (need updating)
        int n_un_unsat,
        int n_sat,
        int n_unsat
    )
    {
        int un_unsat_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (un_unsat_idx >= n_un_unsat) return;
        
        float min_dist = 3.402823466e+38f; // FLT_MAX
        int closest_sat_idx = -1;
        
        // Get position of this unresolved unsaturated node
        float unsat_x = spatial_un_unsat[un_unsat_idx * 2];
        float unsat_y = spatial_un_unsat[un_unsat_idx * 2 + 1];
        
        // Find closest saturated node using L1 distance (Manhattan distance)
        for (int sat_idx = 0; sat_idx < n_sat; sat_idx++) {
            float sat_x = spatial_sat[sat_idx * 2];
            float sat_y = spatial_sat[sat_idx * 2 + 1];
            
            // L1 distance
            float dist = fabsf(unsat_x - sat_x) + fabsf(unsat_y - sat_y);
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_sat_idx = sat_indices[sat_idx];
            }
        }
        
        // Find the position in the original nearest_sat array that corresponds to this un_unsat node
        // and update it with the closest saturated node
        int update_count = 0;
        for (int i = 0; i < n_unsat; i++) {
            if (un_unsat_mask[i]) {  // This position needs updating
                if (update_count == un_unsat_idx) {
                    nearest_sat_indices[i] = (float)closest_sat_idx;
                    break;
                }
                update_count++;
            }
        }
    }
    
} // extern "C"