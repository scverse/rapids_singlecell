extern "C" {

    __device__ double compute_entropy_device(
        const double* __restrict__ conc,
        const int* __restrict__ sat,
        int n_sat
    ) {
        double total_sum = 0.0;
        double entropy = 0.0;
        
        for (int i = 0; i < n_sat; i++) {
            double val = conc[sat[i]];
            if (val > 0.0) {
                total_sum += val;
            }
        }
        
        if (total_sum <= 0.0) return 0.0;
        
        for (int i = 0; i < n_sat; i++) {
            double val = conc[sat[i]];
            if (val > 0.0) {
                double normalized = val / total_sum;
                entropy += -normalized * log(normalized);
            }
        }
        
        return entropy / n_sat;
    }

    __global__ void sepal_simulation(
        const double* __restrict__ gene_data,    // [n_cells] input gene expression
        const int* __restrict__ sat,             // [n_sat] saturated node indices
        const int* __restrict__ sat_idx,         // [n_sat * sat_thresh] neighborhoods
        const int* __restrict__ unsat,           // [n_unsat] unsaturated node indices
        const int* __restrict__ unsat_idx,       // [n_unsat] mapping to saturated
        double* __restrict__ concentration,      // [n_cells] working concentration array (global memory)
        double* __restrict__ derivatives,        // [n_cells] working derivatives array (global memory)
        double* __restrict__ result,             // [1] output
        int n_cells,
        int n_sat,
        int n_unsat,
        int sat_thresh,
        int max_neighs,
        int n_iter,
        double dt,
        double thresh,
        bool debug
    )
    {
        int tid = threadIdx.x;
        int blockSize = blockDim.x;
        
        __shared__ bool has_converged;
        __shared__ int convergence_iteration;
        __shared__ double prev_entropy;
        
        if (tid == 0) {
            has_converged = false;
            convergence_iteration = n_iter;
            prev_entropy = 1.0;
            
            if (debug) {
                printf("=== GPU DEBUG (FIXED ALGO) ===\n");
                printf("n_cells: %d, n_sat: %d, n_unsat: %d\n", n_cells, n_sat, n_unsat);
            }
        }
        
        // Initialize concentration array in parallel
        for (int i = tid; i < n_cells; i += blockSize) {
            concentration[i] = gene_data[i];
            derivatives[i] = 0.0;
        }
        __syncthreads();
        
        const double D = 1.0;
        const double h = 1.0;
        
        // Debug initial state
        if (debug && tid == 0) {
            printf("Initial conc[0:5]: %.6f %.6f %.6f %.6f %.6f\n", 
                   concentration[0], concentration[1], concentration[2], 
                   concentration[3], concentration[4]);
            printf("sat[0:3]: %d %d %d\n", sat[0], sat[1], sat[2]);
        }
        
        for (int iter = 0; iter < n_iter; iter++) {
            
            // Phase 1: Compute derivatives for saturated nodes ONLY
            // This matches CPU: dcdt[sat] = D * d2 where d2 = laplacian(conc[sat], nhood)
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];           // actual node index in concentration array
                
                // Compute neighborhood sum for this saturated node
                double neighbor_sum = 0.0;
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];
                    neighbor_sum += concentration[neighbor_idx];
                }
                
                double center = concentration[sat_node_idx];
                double d2;
                
                // Apply correct Laplacian (matches CPU exactly)
                if (max_neighs == 4) {
                    // CPU: _laplacian_rect: (nbrs - 4 * centers) / h²
                    d2 = (neighbor_sum - 4.0 * center) / (h * h);
                } else if (max_neighs == 6) {
                    // CPU: _laplacian_hex: ((nbrs - 6 * centers) / h²) * (2/3)  
                    d2 = (2.0 * (neighbor_sum - 6.0 * center) / (3.0 * h * h));
                }
                
                // Store derivative: dcdt[sat] = D * d2
                derivatives[sat_node_idx] = D * d2;
            }
            
            __syncthreads(); // Wait for all derivatives to be computed
            
            // Phase 2: Update saturated concentrations
            // CPU: conc[sat] += dcdt[sat] * dt
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];
                concentration[sat_node_idx] += derivatives[sat_node_idx] * dt;
                concentration[sat_node_idx] = fmax(0.0, concentration[sat_node_idx]);
            }
            
            // Phase 3: Update unsaturated concentrations  
            // CPU: conc[unsat] += dcdt[unsat_idx] * dt
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_node_idx = unsat[i];       // actual unsaturated node index
                int mapped_sat_idx = unsat_idx[i];   // corresponding saturated node index
                concentration[unsat_node_idx] += derivatives[mapped_sat_idx] * dt;
                concentration[unsat_node_idx] = fmax(0.0, concentration[unsat_node_idx]);
            }
            
            __syncthreads(); // Wait for all updates
            
            // Debug iteration details for first few iterations
            if (debug && iter < 5 && tid == 0) {
                printf("Iter %d:\n", iter);
                printf("  neighbor_sum[0]: computed in kernel\n");
                printf("  conc[sat[0:3]]: %.6f %.6f %.6f\n",
                       concentration[sat[0]], concentration[sat[1]], concentration[sat[2]]);
                printf("  dcdt[sat[0:3]]: %.6f %.6f %.6f\n",
                       derivatives[sat[0]], derivatives[sat[1]], derivatives[sat[2]]);
            }
            
            // Phase 4: Compute entropy and check convergence (single thread)
            if (tid == 0) {
                double entropy = compute_entropy_device(concentration, sat, n_sat);
                double entropy_diff = fabs(entropy - prev_entropy);
                
                if (debug && iter < 5) {
                    printf("  entropy: %.6f, diff: %.6f\n", entropy, entropy_diff);
                }
                
                if (entropy_diff <= thresh) {
                    has_converged = true;
                    convergence_iteration = iter;
                    
                    if (debug) {
                        printf("  GPU CONVERGED at iteration %d\n", iter);
                    }
                }
                
                prev_entropy = entropy;
            }
            
            __syncthreads();
            
            if (has_converged) {
                break;
            }
        }
        
        if (tid == 0) {
            result[0] = has_converged ? dt * convergence_iteration : -999999.0;
            
            if (debug) {
                printf("GPU Final result: %.6f\n", result[0]);
            }
        }
    }

} // extern "C"