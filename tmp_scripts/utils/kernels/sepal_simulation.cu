extern "C" {

    __device__ double compute_entropy_device(
        const double* __restrict__ conc,
        int n_sat  // Only need n_sat since saturated nodes are indices 0 to n_sat-1
    ) {
        double total_sum = 0.0;
        double entropy = 0.0;
        
        // Sequential access: saturated nodes are indices 0 to n_sat-1
        for (int i = 0; i < n_sat; i++) {
            double val = conc[i];  // Direct sequential access!
            if (val > 0.0) {
                total_sum += val;
            }
        }
        
        if (total_sum <= 0.0) return 0.0;
        
        for (int i = 0; i < n_sat; i++) {
            double val = conc[i];  // Direct sequential access!
            if (val > 0.0) {
                double normalized = val / total_sum;
                entropy += -normalized * log(normalized);
            }
        }
        
        return entropy;
    }

    __global__ void sepal_simulation(
        const double* __restrict__ vals,         // [n_reordered * n_genes] reordered gene data (sat first, then unsat)
        int gene_idx,                            // which gene to process
        const int* __restrict__ sat_idx,         // [n_sat * sat_thresh] neighborhood indices (remapped to sequential)
        const int* __restrict__ unsat_idx,       // [n_unsat] mapping to saturated (remapped to sequential)
        double* __restrict__ concentration,      // [n_reordered] working concentration array
        double* __restrict__ derivatives,        // [n_reordered] working derivatives array
        float* __restrict__ result,             // [1] output
        int n_reordered,                        // n_sat + n_unsat (total reordered size)
        int n_genes,                            // number of genes
        int n_sat,                              // number of saturated (indices 0 to n_sat-1)
        int n_unsat,                            // number of unsaturated (indices n_sat to n_sat+n_unsat-1)
        int sat_thresh,                         // same as max_neighs
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
        }
        
        // Initialize concentration array in parallel with SEQUENTIAL ACCESS
        for (int i = tid; i < n_reordered; i += blockSize) {
            concentration[i] = vals[i * n_genes + gene_idx];  // Sequential read from reordered data!
            derivatives[i] = 0.0;
        }
        __syncthreads();
        
        const double D = 1.0;
        const double h = 1.0;
        
        for (int iter = 0; iter < n_iter; iter++) {
            
            // Phase 1: Compute derivatives for saturated nodes (SEQUENTIAL ACCESS)
            // Saturated nodes are indices 0 to n_sat-1
            for (int i = tid; i < n_sat; i += blockSize) {
                // i is now the direct index in concentration array (sequential!)
                
                // Compute neighborhood sum for this saturated node
                double neighbor_sum = 0.0;
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];  // Already remapped to sequential indices
                    neighbor_sum += concentration[neighbor_idx];  // Sequential access!
                    
                    if (debug && iter == 0 && i < 5) {
                        printf("GPU iter=%d, sat_idx=%d, neighbor_sum=%.6f, neighbor_idx=%d, neighbor_concentration=%.6f\n", 
                               iter, i, neighbor_sum, neighbor_idx, concentration[neighbor_idx]);
                    }
                }
                
                double center = concentration[i];  // Direct sequential access!
                double d2;
                
                // Apply correct Laplacian (matches CPU exactly)
                if (max_neighs == 4) {
                    d2 = (neighbor_sum - 4.0 * center) / (h * h);
                } else if (max_neighs == 6) {
                    d2 = (2.0 * (neighbor_sum - 6.0 * center) / (3.0 * h * h));
                }
                
                // Store derivative: dcdt[sat] = D * d2
                derivatives[i] = D * d2;  // Sequential write!
            }
            
            __syncthreads(); // Wait for all derivatives to be computed
            
            // Phase 2: Update saturated concentrations (SEQUENTIAL ACCESS)
            for (int i = tid; i < n_sat; i += blockSize) {
                concentration[i] += derivatives[i] * dt;  // Sequential access!
                concentration[i] = fmax(0.0, concentration[i]);
            }
            
            // Phase 3: Update unsaturated concentrations (SEQUENTIAL ACCESS)
            // Unsaturated nodes are indices n_sat to n_sat+n_unsat-1
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_idx_in_array = n_sat + i;  // Sequential index for unsaturated node
                int mapped_sat_idx = unsat_idx[i];   // Corresponding saturated node (already remapped)
                concentration[unsat_idx_in_array] += derivatives[mapped_sat_idx] * dt;  // Sequential access!
                concentration[unsat_idx_in_array] = fmax(0.0, concentration[unsat_idx_in_array]);
            }
            
            __syncthreads(); // Wait for all updates
            
            // Phase 4: Compute entropy and check convergence (single thread)
            if (tid == 0) {
                double entropy = compute_entropy_device(concentration, n_sat);  // Sequential access in entropy!
                entropy = entropy / n_sat;
                double entropy_diff = fabs(entropy - prev_entropy);

                if (entropy_diff <= thresh && !has_converged) {
                    has_converged = true;
                    convergence_iteration = iter;
                }
                
                prev_entropy = entropy;
            }
            
            __syncthreads();
            
            if (has_converged) {
                break;
            }
        }
        
        if (tid == 0) {
            result[0] = has_converged ? (float)convergence_iteration : -999999.0;
        }
    }

} // extern "C"