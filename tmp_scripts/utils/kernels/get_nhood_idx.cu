extern "C" {

    __global__ void get_nhood_idx(
        const int* __restrict__ unsat_nodes,      // [n_unsat] unsaturated node indices
        const int* __restrict__ g_indptr,         // [n_nodes + 1] CSR indptr
        const int* __restrict__ g_indices,        // [nnz] CSR indices
        const bool* __restrict__ sat_mask,        // [n_nodes] boolean mask for saturated nodes
        int* __restrict__ nearest_sat,            // [n_unsat] output (-1 if unresolved)
        int n_unsat
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Only handle unsaturated nodes (find saturated neighbors)  
        if (tid < n_unsat) {
            int node = unsat_nodes[tid];
            int start = g_indptr[node];
            int end = g_indptr[node + 1];
            
            nearest_sat[tid] = -1; // Mark as unresolved initially
            
            // Search neighborhood for ANY saturated node using boolean mask
            for (int i = start; i < end; i++) {
                int neighbor = g_indices[i];
                
                // Use boolean mask for O(1) lookup
                if (sat_mask[neighbor]) {
                    nearest_sat[tid] = neighbor;
                    break;  // Take first saturated neighbor found
                }
            }
            if (nearest_sat[tid] == -1) {
                printf("Unresolved node: %d\n", tid);
            }
        }
    }
    
    } // extern "C"