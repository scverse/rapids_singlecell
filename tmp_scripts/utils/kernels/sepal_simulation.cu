extern "C" {

    __device__ double compute_entropy_device(
        const double* __restrict__ conc,
        const int* __restrict__ sat,             // INDEX ARRAY: saturated node indices
        int n_sat
    ) {
        double total_sum = 0.0;
        double entropy = 0.0;
        
        // Compute total sum of saturated concentrations
        for (int i = 0; i < n_sat; i++) {
            double val = conc[sat[i]];  // sat[i] is INDEX into conc array
            if (val > 0.0) {
                total_sum += val;
            }
        }
        
        if (total_sum <= 0.0) {
            return 0.0;
        }
        
        // Compute entropy
        for (int i = 0; i < n_sat; i++) {
            double val = conc[sat[i]];  // sat[i] is INDEX into conc array
            if (val > 0.0) {
                double normalized = val / total_sum;
                entropy += -normalized * log(normalized);
            }
        }
        
        return entropy / n_sat;
    }

    __global__ void sepal_simulation(
        const double* __restrict__ gene_data,    // [n_cells] input gene expression for ONE gene
        const int* __restrict__ sat,             // [n_sat] INDEX ARRAY: saturated node indices
        const int* __restrict__ sat_idx,         // [n_sat * sat_thresh] neighborhoods (2D flattened)
        const int* __restrict__ unsat,           // [n_unsat] INDEX ARRAY: unsaturated node indices  
        const int* __restrict__ unsat_idx,       // [n_unsat] INDEX ARRAY: mapping unsaturated->saturated indices
        double* __restrict__ result,             // [1] output: dt * convergence_time
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
        // This kernel processes ONE GENE ONLY
        int tid = threadIdx.x;
        int blockSize = blockDim.x;
        
        // Shared memory for this gene's concentration values and working arrays
        extern __shared__ double shared_mem[];
        
        // Calculate offsets explicitly to prevent overlap
        const int conc_offset = 0;
        const int nhood_offset = n_cells;  
        const int dcdt_offset = n_cells + n_sat;
        
        double* concentration = &shared_mem[conc_offset];      // [n_cells]
        double* neighbor_sums = &shared_mem[nhood_offset];     // [n_sat]
        double* derivatives = &shared_mem[dcdt_offset];        // [n_cells]
        
        // Add convergence flag in shared memory
        __shared__ bool has_converged;
        __shared__ int convergence_iteration;
        
        // Initialize convergence flag
        if (tid == 0) {
            has_converged = false;
            convergence_iteration = n_iter;
            
            if (debug) {
                printf("=== GPU DEBUG ===\n");
                printf("n_cells: %d, n_sat: %d, n_unsat: %d\n", n_cells, n_sat, n_unsat);
                printf("Memory layout: conc[0:%d] nhood[%d:%d] dcdt[%d:%d]\n", 
                       n_cells-1, nhood_offset, nhood_offset+n_sat-1, 
                       dcdt_offset, dcdt_offset+n_cells-1);
            }
        }
        
        // Initialize concentration from gene data (parallel across threads)
        for (int i = tid; i < n_cells; i += blockSize) {
            concentration[i] = gene_data[i];
        }
        
        // Initialize working arrays ONCE
        for (int i = tid; i < n_sat; i += blockSize) {
            neighbor_sums[i] = 0.0;
        }
        for (int i = tid; i < n_cells; i += blockSize) {
            derivatives[i] = 0.0;
        }
        
        __syncthreads();
        
        // Debug initial state
        if (debug && tid == 0) {
            printf("Initial conc[0:5]: %.6f %.6f %.6f %.6f %.6f\n", 
                   concentration[0], concentration[1], concentration[2], 
                   concentration[3], concentration[4]);
            printf("sat[0:3]: %d %d %d\n", sat[0], sat[1], sat[2]);
            if (n_unsat > 0) {
                printf("unsat[0:3]: %d %d %d\n", unsat[0], 
                       n_unsat > 1 ? unsat[1] : -1, 
                       n_unsat > 2 ? unsat[2] : -1);
            }
        }
        
        double prev_entropy = 1.0;
        const double D = 1.0;  // Diffusion coefficient (same as CPU)
        const double h = 1.0;  // Grid spacing (weights are 1.0 in CPU)
        
        // Main diffusion loop - equivalent to CPU _diffusion function
        int debug_iterations = debug ? 5 : 0;
        
        for (int iter = 0; iter < n_iter; iter++) {
            
            // Phase 1: Compute neighborhood sums (parallel over saturated nodes)
            for (int i = tid; i < n_sat; i += blockSize) {
                double sum = 0.0;
                // sat_idx is [n_sat, sat_thresh] - access row i
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];  // Row-major access
                    if (debug && neighbor_idx >= 0 && neighbor_idx < n_cells) {
                        sum += concentration[neighbor_idx];
                    } else if (!debug) {
                        sum += concentration[neighbor_idx];
                    }
                }
                neighbor_sums[i] = sum;
            }
            
            __syncthreads();  // REQUIRED: All threads need neighbor_sums[] computed
            
            // Debug neighborhood sums for first few iterations
            if (debug && iter < debug_iterations && tid == 0) {
                printf("Iter %d:\n", iter);
                printf("  nhood[0:3]: %.6f %.6f %.6f\n", 
                       neighbor_sums[0], neighbor_sums[1], neighbor_sums[2]);
                printf("  conc[sat][0:3]: %.6f %.6f %.6f\n",
                       concentration[sat[0]], concentration[sat[1]], concentration[sat[2]]);
            }
            
            // Phase 2: Compute derivatives for saturated nodes ONLY
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];           // sat[i] is INDEX: actual node index  
                
                if (debug && (sat_node_idx < 0 || sat_node_idx >= n_cells)) {
                    continue;  // Skip invalid indices in debug mode
                }
                
                double center = concentration[sat_node_idx];   // Get concentration at that node
                double neighbors = neighbor_sums[i];           // Sum of neighbor concentrations
                double d2;
                
                // Apply Laplacian based on max_neighs - matches CPU exactly
                if (max_neighs == 4) {
                    // _laplacian_rect: (nbrs - 4 * centers) / h²
                    d2 = (neighbors - 4.0 * center) / (h * h);
                } else if (max_neighs == 6) {
                    // _laplacian_hex: ((nbrs - 6 * centers) / h²) * (2/3)
                    d2 = (2.0 * (neighbors - 6.0 * center) / (3.0 * h * h));
                }
                
                // Apply diffusion coefficient D (CPU: dcdt[sat] = D * d2)
                derivatives[sat_node_idx] = D * d2;  // Store at the actual node index
            }
            
            __syncthreads();  // REQUIRED: Phase 3&4 read derivatives[] written by Phase 2
            
            // Debug derivatives for first few iterations
            if (debug && iter < debug_iterations && tid == 0) {
                printf("  d2[0:3]: %.6f %.6f %.6f\n",
                       derivatives[sat[0]], derivatives[sat[1]], derivatives[sat[2]]);
            }
            
            // Phase 3: Update saturated concentrations  
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];
                
                if (debug && (sat_node_idx < 0 || sat_node_idx >= n_cells)) {
                    continue;  // Skip invalid indices in debug mode
                }
                
                concentration[sat_node_idx] += derivatives[sat_node_idx] * dt;
                concentration[sat_node_idx] = fmax(0.0, concentration[sat_node_idx]);  // Clamp to >= 0
            }
            
            // Phase 4: Update unsaturated concentrations
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_node_idx = unsat[i];       // unsat[i] is INDEX: actual node index
                int mapped_sat_idx = unsat_idx[i];   // unsat_idx[i] is INDEX: mapped saturated node
                
                if (debug && (unsat_node_idx < 0 || unsat_node_idx >= n_cells || 
                             mapped_sat_idx < 0 || mapped_sat_idx >= n_cells)) {
                    continue;  // Skip invalid indices in debug mode
                }
                
                concentration[unsat_node_idx] += derivatives[mapped_sat_idx] * dt;  // Use derivative from mapped saturated node
                concentration[unsat_node_idx] = fmax(0.0, concentration[unsat_node_idx]);  // Clamp to >= 0
            }
            
            __syncthreads();  // REQUIRED: Phase 5 reads concentration[] written by Phase 3&4
            
            // Debug updated concentrations for first few iterations
            if (debug && iter < debug_iterations && tid == 0) {
                printf("  After update conc[sat][0:3]: %.6f %.6f %.6f\n",
                       concentration[sat[0]], concentration[sat[1]], concentration[sat[2]]);
                if (n_unsat > 0) {
                    printf("  After update conc[unsat][0:3]: %.6f %.6f %.6f\n",
                           concentration[unsat[0]], 
                           n_unsat > 1 ? concentration[unsat[1]] : 0.0,
                           n_unsat > 2 ? concentration[unsat[2]] : 0.0);
                }
            }
            
            // Phase 5: Compute entropy and check convergence (single thread)
            if (tid == 0) {
                double entropy = compute_entropy_device(concentration, sat, n_sat);  
                double entropy_diff = fabs(entropy - prev_entropy);
                
                if (debug && iter < debug_iterations) {
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
            
            __syncthreads();  // REQUIRED: All threads need to see convergence flag
            
            // Clean convergence break
            if (has_converged) {
                break;  // All threads exit cleanly
            }
        }
        
        // Write result (convergence time * dt, or -999999 for no convergence)
        if (tid == 0) {
            if (has_converged) {
                result[0] = dt * convergence_iteration;  // Same as CPU: dt * iteration_number
                
                if (debug) {
                    printf("GPU Final result: %.6f\n", result[0]);
                }
            } else {
                result[0] = -999999.0;  // No convergence
                
                if (debug) {
                    printf("GPU No convergence\n");
                }
            }
        }
    }

} // extern "C"