extern "C" {

    __global__ void get_nhood_idx_with_distance(
        const int* __restrict__ unsat_nodes,      // [n_unsat] unsaturated node indices
        const float* __restrict__ spatial,        // [n_nodes * 2] spatial coordinates (x,y)
        const int* __restrict__ sat_nodes,        // [n_sat] saturated node indices
        const int* __restrict__ g_indptr,         // [n_nodes + 1] CSR indptr
        const int* __restrict__ g_indices,        // [nnz] CSR indices
        const bool* __restrict__ sat_mask,        // [n_nodes] boolean mask for saturated nodes
        int* __restrict__ nearest_sat,            // [n_unsat] output
        int n_unsat,
        int n_sat
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n_unsat) return;
        
        int node = unsat_nodes[tid];
        float node_x = spatial[node * 2];
        float node_y = spatial[node * 2 + 1];
        
        float min_dist = -1.0f;  // -1.0f means no closest sat found yet
        int closest = -1;
        
        // Phase 1: Check graph neighbors for saturated nodes
        int start = g_indptr[node];
        int end = g_indptr[node + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = g_indices[i];
            if (sat_mask[neighbor]) {
                closest = neighbor;  // Take first
                break;               // Stop immediately
            }
        }
        
        // Phase 2: If no saturated graph neighbors, search ALL saturated nodes
        if (closest == -1) {
            for (int i = 0; i < n_sat; i++) {
                int sat_node = sat_nodes[i];
                float sat_x = spatial[sat_node * 2];
                float sat_y = spatial[sat_node * 2 + 1];
                float dist = fabsf(node_x - sat_x) + fabsf(node_y - sat_y);
                
                if (min_dist < 0.0f || dist < min_dist) {
                    min_dist = dist;
                    closest = sat_node;
                }
            }
        }
        
        nearest_sat[tid] = closest;
    }
    
    } // extern "C"