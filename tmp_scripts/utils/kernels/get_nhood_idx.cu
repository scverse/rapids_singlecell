extern "C" {
__global__ void get_nhood_idx(
    const int* __restrict__ sat_nodes,        // [n_sat] 
    const int* __restrict__ unsat_nodes,      // [n_unsat]
    const int* __restrict__ g_indptr,        // [n_nodes + 1]
    const int* __restrict__ g_indices,       // [nnz]
    int* __restrict__ sat_neighborhoods,     // [n_sat * sat_thresh] pre-allocated
    int* __restrict__ nearest_sat,           // [n_unsat] pre-allocated (-1 = unresolved)
    int n_sat,
    int n_unsat, 
    int sat_thresh
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle saturated nodes (extract neighborhoods)
    if (tid < n_sat) {
        int node = sat_nodes[tid];
        int start = g_indptr[node];
        
        // Copy neighborhood directly to pre-allocated output
        for (int i = 0; i < sat_thresh; i++) {
            sat_neighborhoods[tid * sat_thresh + i] = g_indices[start + i];
        }
    }
    
    // Handle unsaturated nodes (find saturated neighbors)  
    if (tid < n_unsat) {
        int node = unsat_nodes[tid];
        int start = g_indptr[node];
        int end = g_indptr[node + 1];
        
        nearest_sat[tid] = -1; // Mark as unresolved initially
        
        // Search neighborhood for ANY saturated node
        for (int i = start; i < end; i++) {
            int neighbor = g_indices[i];
            
            // Check if neighbor is in sat_nodes (binary search or hash lookup)
            if (is_saturated(neighbor, sat_nodes, n_sat)) {
                nearest_sat[tid] = neighbor;
                break;
            }
        }
    }
}
} // extern "C"