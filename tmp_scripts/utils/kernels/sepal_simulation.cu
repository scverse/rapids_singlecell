extern "C" {

    // Thread-safe entropy computation using parallel reduction
    __device__ double compute_entropy_parallel(
        const double* __restrict__ conc,
        int n_sat,
        int tid,
        int blockSize
    ) {
        __shared__ double total_sum_shared[256];  // Max block size
        __shared__ double entropy_shared[256];
        
        // Phase 1: Parallel computation of total sum
        double local_sum = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[i];
            if (val > 0.0) {
                local_sum += val;
            }
        }
        
        total_sum_shared[tid] = local_sum;
        __syncthreads();
        
        // Reduction for total sum
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                total_sum_shared[tid] += total_sum_shared[tid + s];
            }
            __syncthreads();
        }
        
        double total_sum = total_sum_shared[0];
        if (total_sum <= 0.0) return 0.0;
        
        // Phase 2: Parallel computation of entropy
        double local_entropy = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[i];
            if (val > 0.0) {
                double normalized = val / total_sum;
                local_entropy += -normalized * log(normalized);
            }
        }
        
        entropy_shared[tid] = local_entropy;
        __syncthreads();
        
        // Reduction for entropy
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                entropy_shared[tid] += entropy_shared[tid + s];
            }
            __syncthreads();
        }
        
        return entropy_shared[0];
    }

    __global__ void sepal_simulation(
        const double* __restrict__ vals,         // [n_reordered * n_genes] reordered gene data
        int gene_idx,                            // which gene to process
        const int* __restrict__ sat_idx,         // [n_sat * sat_thresh] neighborhood indices
        const int* __restrict__ unsat_idx,       // [n_unsat] mapping to saturated
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
        
        // Shared variables for convergence (accessed by all threads)
        __shared__ bool has_converged;
        __shared__ int convergence_iteration;
        __shared__ double prev_entropy;
        
        // Initialize shared variables (all threads participate)
        if (tid == 0) {
            has_converged = false;
            convergence_iteration = n_iter;
            prev_entropy = 1.0;
        }
        __syncthreads();  // ALL threads wait here
        
        // Initialize concentration array in parallel (ALL threads participate)
        for (int i = tid; i < n_reordered; i += blockSize) {
            concentration[i] = vals[i * n_genes + gene_idx];
            derivatives[i] = 0.0;
        }
        __syncthreads();  // ALL threads wait here
        
        const double D = 1.0;
        const double h = 1.0;
        
        // Main simulation loop
        for (int iter = 0; iter < n_iter; iter++) {
            
            // Phase 1: Compute derivatives for saturated nodes (ALL threads can participate)
            for (int i = tid; i < n_sat; i += blockSize) {
                double neighbor_sum = 0.0;
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];
                    neighbor_sum += concentration[neighbor_idx];
                }
                
                double center = concentration[i];
                double d2;
                
                if (max_neighs == 4) {
                    d2 = (neighbor_sum - 4.0 * center) / (h * h);
                } else if (max_neighs == 6) {
                    d2 = (2.0 * (neighbor_sum - 6.0 * center) / (3.0 * h * h));
                }
                
                derivatives[i] = D * d2;
            }
            __syncthreads();  // ALL threads wait here
            
            // Phase 2: Update saturated concentrations (ALL threads can participate)
            for (int i = tid; i < n_sat; i += blockSize) {
                concentration[i] += derivatives[i] * dt;
                concentration[i] = fmax(0.0, concentration[i]);
            }
            
            // Phase 3: Update unsaturated concentrations (ALL threads can participate)
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_idx_in_array = n_sat + i;
                int mapped_sat_idx = unsat_idx[i];
                concentration[unsat_idx_in_array] += derivatives[mapped_sat_idx] * dt;
                concentration[unsat_idx_in_array] = fmax(0.0, concentration[unsat_idx_in_array]);
            }
            __syncthreads();  // ALL threads wait here
            
            // Phase 4: Convergence check (ALL threads participate in entropy calculation)
            double entropy = compute_entropy_parallel(concentration, n_sat, tid, blockSize);
            
            // Only thread 0 updates convergence status, but ALL threads read it
            if (tid == 0) {
                entropy = entropy / n_sat;
                double entropy_diff = fabs(entropy - prev_entropy);

                if (entropy_diff <= thresh && !has_converged) {
                    has_converged = true;
                    convergence_iteration = iter;
                }
                
                prev_entropy = entropy;
            }
            __syncthreads();  // ALL threads wait here and read updated has_converged
            
            // ALL threads check convergence status together
            if (has_converged) {
                break;  // ALL threads break together
            }
        }
        
        // Only thread 0 writes result (no synchronization needed after loop)
        if (tid == 0) {
            result[0] = has_converged ? (float)convergence_iteration : -999999.0;
        }
    }

} // extern "C"