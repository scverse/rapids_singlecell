extern "C" {

    __device__ float compute_entropy_device(
        const float* __restrict__ conc,
        const int* __restrict__ sat,             // INDEX ARRAY: saturated node indices
        int n_sat
    ) {
        float total_sum = 0.0f;
        float entropy = 0.0f;
        
        // Compute total sum of saturated concentrations
        for (int i = 0; i < n_sat; i++) {
            float val = conc[sat[i]];  // sat[i] is INDEX into conc array
            if (val > 0.0f) {
                total_sum += val;
            }
        }
        
        if (total_sum <= 0.0f) {
            return 0.0f;
        }
        
        // Compute entropy
        for (int i = 0; i < n_sat; i++) {
            float val = conc[sat[i]];  // sat[i] is INDEX into conc array
            if (val > 0.0f) {
                float normalized = val / total_sum;
                entropy += -normalized * logf(normalized);
            }
        }
        
        return entropy / n_sat;
    }

    __global__ void sepal_simulation(
        const float* __restrict__ gene_data,     // [n_cells] input gene expression for ONE gene
        const int* __restrict__ sat,             // [n_sat] INDEX ARRAY: saturated node indices
        const int* __restrict__ sat_idx,         // [n_sat * sat_thresh] neighborhoods (2D flattened)
        const int* __restrict__ unsat,           // [n_unsat] INDEX ARRAY: unsaturated node indices  
        const int* __restrict__ unsat_idx,       // [n_unsat] INDEX ARRAY: mapping unsaturated->saturated indices
        float* __restrict__ result,              // [1] output: dt * convergence_time
        int n_cells,
        int n_sat,
        int n_unsat,
        int sat_thresh,
        int max_neighs,
        int n_iter,
        float dt,
        float thresh
    )
    {
        // This kernel processes ONE GENE ONLY
        int tid = threadIdx.x;
        int blockSize = blockDim.x;
        
        // Shared memory for this gene's concentration values and working arrays
        extern __shared__ float shared_mem[];
        float* conc = shared_mem;                                      // [n_cells]
        float* nhood = &conc[n_cells];                                // [n_sat]
        float* dcdt = &nhood[n_sat];                                  // [n_cells]
        
        // Add convergence flag in shared memory
        __shared__ bool converged;
        __shared__ int conv_iter;
        
        // Initialize convergence flag
        if (tid == 0) {
            converged = false;
            conv_iter = n_iter;
        }
        
        // Initialize concentration from gene data (parallel across threads)
        for (int i = tid; i < n_cells; i += blockSize) {
            conc[i] = gene_data[i];
        }
        
        // Initialize working arrays ONCE (like CPU: dcdt = np.zeros(conc_shape) outside loop)
        for (int i = tid; i < n_sat; i += blockSize) {
            nhood[i] = 0.0f;
        }
        for (int i = tid; i < n_cells; i += blockSize) {
            dcdt[i] = 0.0f;
        }
        
        __syncthreads();
        
        float prev_ent = 1.0f;
        const float D = 1.0f;  // Diffusion coefficient (same as CPU)
        const float h = 1.0f;  // Grid spacing (weights are 1.0 in CPU)
        
        // Main diffusion loop - equivalent to CPU _diffusion function
        for (int iter = 0; iter < n_iter; iter++) {
            
            // Phase 1: Compute neighborhood sums (parallel over saturated nodes)
            // CPU equivalent: for j in range(sat_shape): nhood[j] = np.sum(conc[sat_idx[j]])
            for (int i = tid; i < n_sat; i += blockSize) {
                float sum = 0.0f;
                // sat_idx is [n_sat, sat_thresh] - access row i
                for (int j = 0; j < sat_thresh; j++) {
                    int neighbor_idx = sat_idx[i * sat_thresh + j];  // Row-major access
                    sum += conc[neighbor_idx];
                }
                nhood[i] = sum;
            }
            
            __syncthreads();  // REQUIRED: All threads need nhood[] computed
            
            // Phase 2: Compute derivatives for saturated nodes ONLY
            // CPU equivalent: dcdt[sat] = D * d2
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];           // sat[i] is INDEX: actual node index  
                float center = conc[sat_node_idx];   // Get concentration at that node
                float neighbors = nhood[i];          // Sum of neighbor concentrations
                float d2;
                
                // Apply Laplacian based on max_neighs - matches CPU exactly
                if (max_neighs == 4) {
                    // _laplacian_rect: (nbrs - 4 * centers) / h²
                    d2 = (neighbors - 4.0f * center) / (h * h);
                } else if (max_neighs == 6) {
                    // _laplacian_hex: ((nbrs - 6 * centers) / h²) * (2/3)
                    d2 = (2.0f * (neighbors - 6.0f * center) / (3.0f * h * h));
                }
                
                // Apply diffusion coefficient D (CPU: dcdt[sat] = D * d2)
                dcdt[sat_node_idx] = D * d2;  // Store at the actual node index
            }
            
            __syncthreads();  // REQUIRED: Phase 3&4 read dcdt[] written by Phase 2
            
            // Phase 3: Update saturated concentrations  
            // CPU equivalent: conc[sat] += dcdt[sat] * dt
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_node_idx = sat[i];
                conc[sat_node_idx] += dcdt[sat_node_idx] * dt;
                conc[sat_node_idx] = fmaxf(0.0f, conc[sat_node_idx]);  // Clamp to >= 0
            }
            
            // Phase 4: Update unsaturated concentrations
            // CPU equivalent: conc[unsat] += dcdt[unsat_idx] * dt  
            // NOTE: unsat_idx[i] points to saturated node indices, so dcdt[unsat_idx[i]] gives the derivative
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_node_idx = unsat[i];       // unsat[i] is INDEX: actual node index
                int mapped_sat_idx = unsat_idx[i];   // unsat_idx[i] is INDEX: mapped saturated node
                conc[unsat_node_idx] += dcdt[mapped_sat_idx] * dt;  // Use derivative from mapped saturated node
                conc[unsat_node_idx] = fmaxf(0.0f, conc[unsat_node_idx]);  // Clamp to >= 0
            }
            
            __syncthreads();  // REQUIRED: Phase 5 reads conc[] written by Phase 3&4
            
            // Phase 5: Compute entropy and check convergence (single thread)
            // CPU equivalent: ent = _entropy(conc[sat]) / sat_shape; entropy_arr[i] = np.abs(ent - prev_ent)
            if (tid == 0) {
                float ent = compute_entropy_device(conc, sat, n_sat);  
                float entropy_diff = fabsf(ent - prev_ent);
                
                if (entropy_diff <= thresh) {
                    converged = true;
                    conv_iter = iter;
                }
                
                prev_ent = ent;
            }
            
            __syncthreads();  // REQUIRED: All threads need to see convergence flag
            
            // Clean convergence break
            if (converged) {
                break;  // All threads exit cleanly
            }
        }
        
        // Write result (convergence time * dt, or -999999 for no convergence)
        // CPU equivalent: return float(tmp[0] if len(tmp) else np.nan)
        if (tid == 0) {
            if (converged) {
                result[0] = dt * conv_iter;  // Same as CPU: dt * iteration_number
            } else {
                result[0] = -999999.0f;  // No convergence
            }
        }
    }

} // extern "C"