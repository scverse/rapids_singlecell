extern "C" {
    
    __global__ void sepal_simulation_debug(
        const float* __restrict__ gene_data,
        const int* __restrict__ sat,
        const int* __restrict__ sat_idx,
        const int* __restrict__ unsat,
        const int* __restrict__ unsat_idx,
        float* __restrict__ result,
        float* __restrict__ debug_output,    // [debug_size] debug array
        int n_cells, int n_sat, int n_unsat, int sat_thresh,
        int max_neighs, int n_iter, float dt, float thresh
    )
    {
        int tid = threadIdx.x;
        int blockSize = blockDim.x;
        
        // Shared memory layout
        extern __shared__ float shared_mem[];
        const int conc_offset = 0;
        const int nhood_offset = n_cells;
        const int dcdt_offset = n_cells + n_sat;
        
        float* concentration = &shared_mem[conc_offset];
        float* neighbor_sums = &shared_mem[nhood_offset];
        float* derivatives = &shared_mem[dcdt_offset];
        
        __shared__ bool has_converged;
        __shared__ int convergence_iteration;
        
        if (tid == 0) {
            has_converged = false;
            convergence_iteration = n_iter;
            
            printf("=== GPU DEBUG ===\n");
            printf("n_cells: %d, n_sat: %d, n_unsat: %d\n", n_cells, n_sat, n_unsat);
        }
        
        // Initialize
        for (int i = tid; i < n_cells; i += blockSize) {
            concentration[i] = gene_data[i];
        }
        for (int i = tid; i < n_sat; i += blockSize) {
            neighbor_sums[i] = 0.0f;
        }
        for (int i = tid; i < n_cells; i += blockSize) {
            derivatives[i] = 0.0f;
        }
        
        __syncthreads();
        
        // Debug initial state
        if (tid == 0) {
            printf("Initial conc[0:5]: %.6f %.6f %.6f %.6f %.6f\n", 
                   concentration[0], concentration[1], concentration[2], 
                   concentration[3], concentration[4]);
            printf("sat[0:3]: %d %d %d\n", sat[0], sat[1], sat[2]);
            printf("unsat[0:3]: %d %d %d\n", unsat[0], unsat[1], unsat[2]);
        }
        
        float prev_entropy = 1.0f;
        const float D = 1.0f;
        const float h = 1.0f;
        
        // Main simulation loop - debug first 5 iterations
        for (int iter = 0; iter < min(5, n_iter); iter++) {
            
            // Phase 1: Neighborhood sums
            for (int i = tid; i < n_sat; i += blockSize) {
                float sum = 0.0f;
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];
                    sum += concentration[neighbor_idx];
                }
                neighbor_sums[i] = sum;
            }
            
            __syncthreads();
            
            // Debug neighborhood sums
            if (tid == 0) {
                printf("Iter %d:\n", iter);
                printf("  nhood[0:3]: %.6f %.6f %.6f\n", 
                       neighbor_sums[0], neighbor_sums[1], neighbor_sums[2]);
                printf("  conc[sat][0:3]: %.6f %.6f %.6f\n",
                       concentration[sat[0]], concentration[sat[1]], concentration[sat[2]]);
            }
            
            // Phase 2: Derivatives
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];
                float center = concentration[sat_node_idx];
                float neighbors = neighbor_sums[i];
                float d2;
                
                if (max_neighs == 4) {
                    d2 = (neighbors - 4.0f * center) / (h * h);
                } else if (max_neighs == 6) {
                    d2 = (2.0f * (neighbors - 6.0f * center) / (3.0f * h * h));
                }
                
                derivatives[sat_node_idx] = D * d2;
            }
            
            __syncthreads();
            
            // Debug derivatives
            if (tid == 0) {
                printf("  d2[0:3]: %.6f %.6f %.6f\n",
                       derivatives[sat[0]], derivatives[sat[1]], derivatives[sat[2]]);
            }
            
            // Phase 3: Update saturated
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];
                concentration[sat_node_idx] += derivatives[sat_node_idx] * dt;
                concentration[sat_node_idx] = fmaxf(0.0f, concentration[sat_node_idx]);
            }
            
            // Phase 4: Update unsaturated
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_node_idx = unsat[i];
                int mapped_sat_idx = unsat_idx[i];
                concentration[unsat_node_idx] += derivatives[mapped_sat_idx] * dt;
                concentration[unsat_node_idx] = fmaxf(0.0f, concentration[unsat_node_idx]);
            }
            
            __syncthreads();
            
            // Debug updated concentrations
            if (tid == 0) {
                printf("  After update conc[sat][0:3]: %.6f %.6f %.6f\n",
                       concentration[sat[0]], concentration[sat[1]], concentration[sat[2]]);
                printf("  After update conc[unsat][0:3]: %.6f %.6f %.6f\n",
                       concentration[unsat[0]], concentration[unsat[1]], concentration[unsat[2]]);
            }
            
            // Phase 5: Check convergence
            if (tid == 0) {
                float entropy = compute_entropy_device(concentration, sat, n_sat);
                float entropy_diff = fabsf(entropy - prev_entropy);
                
                printf("  entropy: %.6f, diff: %.6f\n", entropy, entropy_diff);
                
                if (entropy_diff <= thresh) {
                    has_converged = true;
                    convergence_iteration = iter;
                    printf("  GPU CONVERGED at iteration %d\n", iter);
                }
                
                prev_entropy = entropy;
            }
            
            __syncthreads();
            
            if (has_converged) {
                break;
            }
        }
        
        // Continue without debug prints for remaining iterations
        for (int iter = 5; iter < n_iter; iter++) {
            // ... normal simulation loop without prints
        }
        
        // Write result
        if (tid == 0) {
            if (has_converged) {
                result[0] = dt * convergence_iteration;
                printf("GPU Final result: %.6f\n", result[0]);
            } else {
                result[0] = -999999.0f;
                printf("GPU No convergence\n");
            }
        }
    }
}