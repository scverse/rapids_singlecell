extern "C" {

    // **DEADLOCK-FREE ENTROPY COMPUTATION**
    // All threads participate to avoid deadlocks
    __device__ double compute_entropy_cooperative(
        const double* __restrict__ conc,
        int n_sat,
        int tid,
        int blockSize
    ) {
        __shared__ double total_sum_shared[256];
        __shared__ double entropy_shared[256];
        
        // **ALL THREADS PARTICIPATE - Phase 1: Compute total sum**
        double local_sum = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[i];
            if (val > 0.0) {
                local_sum += val;
            }
        }
        
        total_sum_shared[tid] = local_sum;
        __syncthreads();  // SAFE: All threads participate
        
        // **WARP-OPTIMIZED REDUCTION**
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                total_sum_shared[tid] += total_sum_shared[tid + s];
            }
            __syncthreads();  // SAFE: All threads participate
        }
        
        double total_sum = total_sum_shared[0];
        if (total_sum <= 0.0) return 0.0;
        
        // **ALL THREADS PARTICIPATE - Phase 2: Compute entropy**
        double local_entropy = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[i];
            if (val > 0.0) {
                double normalized = val / total_sum;
                local_entropy += -normalized * log(normalized);
            }
        }
        
        entropy_shared[tid] = local_entropy;
        __syncthreads();  // SAFE: All threads participate
        
        // **WARP-OPTIMIZED REDUCTION**
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                entropy_shared[tid] += entropy_shared[tid + s];
            }
            __syncthreads();  // SAFE: All threads participate
        }
        
        return entropy_shared[0] / (double)(n_sat);  // Normalize like CPU
    }

    // **DEADLOCK-FREE MULTI-GENE KERNEL**
    __global__ void sepal_simulation(
        double* __restrict__ concentration_all,  // [n_genes * n_cells] - all genes
        double* __restrict__ derivatives_all,          // [n_genes * n_cells] - all derivatives  
        const int* __restrict__ sat_idx,               // [n_sat * sat_thresh] - flattened
        const int* __restrict__ unsat_idx,             // [n_unsat] - unsat mappings
        float* __restrict__ results,                   // [n_genes] - output for all genes
        int n_cells,                                   // Total cells (can be 1M+)
        int n_genes,                                   // Total genes 
        int n_sat,
        int n_unsat,
        int max_neighs,
        int sat_thresh,
        int n_iter,
        float dt,
        float thresh,
        bool debug
    ) {
        // **EACH BLOCK PROCESSES ONE GENE**
        int gene_idx = blockIdx.x;  // Gene index (0 to n_genes-1)
        int tid = threadIdx.x;      // Thread within block (0 to 255)
        int blockSize = blockDim.x; // 256
        
        if (gene_idx >= n_genes) return;
        
        // **GLOBAL MEMORY POINTERS FOR THIS GENE**
        double* concentration = &concentration_all[gene_idx * n_cells];
        double* derivatives = &derivatives_all[gene_idx * n_cells];
        
        // **CONVERGENCE TRACKING - SHARED BY ALL THREADS**
        __shared__ double prev_entropy;
        __shared__ int convergence_iter;
        __shared__ bool converged_flag;  // **FIX: Shared convergence flag**
        
        if (tid == 0) {
            prev_entropy = 1.0;
            convergence_iter = -1;
            converged_flag = false;
        }
        __syncthreads();  // **ALL THREADS WAIT**
        
        // **ITERATION LOOP**
        for (int iter = 0; iter < n_iter; iter++) {
            
            // **PHASE 1: Compute derivatives for saturated nodes**
            for (int i = tid; i < n_sat; i += blockSize) {
                double neighbor_sum = 0.0;
                
                // **DEBUG: Print details for first few iterations of first genes**
                bool should_debug = debug && gene_idx < 2 && iter < 3 && i < 3;
                
                // Sum neighbors for this saturated node
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];
                    double neighbor_val = concentration[neighbor_idx];
                    neighbor_sum += neighbor_val;
                    
                    if (should_debug) {
                        printf("Gene %d, iter %d, sat_node %d, neighbor %d: idx=%d, val=%e\n", 
                               gene_idx, iter, i, j, neighbor_idx, neighbor_val);
                    }
                }
                
                if (should_debug) {
                    printf("Gene %d, iter %d, sat_node %d: center=%e, neighbor_sum=%e\n", 
                           gene_idx, iter, i, concentration[i], neighbor_sum);
                }
                
                double center = concentration[i];
                double d2;
                double h = 1.0;  // **SPATIAL STEP SIZE**
                if (max_neighs == 4) {
                    d2 = (neighbor_sum - 4.0 * center) / (h * h);  // **FIX: Use h, not dt**
                } else if (max_neighs == 6) {
                    d2 = (neighbor_sum - 6.0 * center) / (h * h);  // **FIX: Use h, not dt**
                    d2 = (d2 * 2.0) / 3.0;  // **MATCH CPU EXACTLY**
                } else {
                    d2 = 0.0;
                }
                derivatives[i] = 1.0 * d2;  // D = 1.0
            }
            __syncthreads();  // **FIX: Ensure derivatives are ready for Phase 3**
            
            // **PHASE 2: Update saturated nodes**
            for (int i = tid; i < n_sat; i += blockSize) {
                concentration[i] += derivatives[i] * dt;
                concentration[i] = fmax(0.0, concentration[i]);
            }
            __syncthreads();  // **REQUIRED: Ensure sat updates complete**
            
            // **PHASE 3: Update unsaturated nodes**
            for (int i = tid; i < n_unsat; i += blockSize) {
                int mapped_sat_idx = unsat_idx[i];
                int unsat_global_idx = n_sat + i;
                concentration[unsat_global_idx] += derivatives[mapped_sat_idx] * dt;
                concentration[unsat_global_idx] = fmax(0.0, concentration[unsat_global_idx]);
            }
            __syncthreads();  // **REQUIRED: Ensure all updates complete**
            
            // **CONVERGENCE CHECK - ALL THREADS PARTICIPATE**
            if (!converged_flag) {  // Check every 10 iterations for efficiency
                // **ALL THREADS CALL ENTROPY FUNCTION**
                double current_entropy = compute_entropy_cooperative(concentration, n_sat, tid, blockSize);
                
                // **ONLY THREAD 0 UPDATES CONVERGENCE STATE**
                if (tid == 0) {
                    double entropy_diff = fabs(current_entropy - prev_entropy);
                    
                    if (entropy_diff <= thresh && convergence_iter == -1) {
                        convergence_iter = iter;
                        converged_flag = true;  // **SET SHARED FLAG**
                    }
                    prev_entropy = current_entropy;
                    
                    // if (debug && gene_idx < 3) {
                    //     printf("Gene %d, iter %d: entropy_diff = %e, thresh = %e\n", 
                    //            gene_idx, iter, entropy_diff, thresh);
                    // }
                }
                __syncthreads();  // **ALL THREADS SEE CONVERGENCE UPDATE**
            }
            
            // **COORDINATED EARLY EXIT - ALL THREADS DECIDE TOGETHER**
            if (converged_flag) {
                break;  // **ALL THREADS EXIT TOGETHER**
            }
        }
        
        // **FINAL RESULT - ONLY THREAD 0 WRITES**
        if (tid == 0) {
            if (convergence_iter >= 0) {
                results[gene_idx] = (float)convergence_iter;
            } else {
                results[gene_idx] = -999999.0;  // No convergence
            }
            
            // if (debug && gene_idx < 3) {
            //     printf("Gene %d: final convergence_iter = %d\n", gene_idx, convergence_iter);
            // }
        }
    }

}  // extern "C"