extern "C" {
    // Computes entropy using cooperative thread reduction
    // Shared memory per block: 4096 bytes (2 arrays of 256 doubles)
    __device__ double compute_entropy_cooperative(
        const double* __restrict__ conc,     // Input concentration values
        int n_sat,                          // Number of saturated nodes
        int tid,                            // Thread ID within block
        int blockSize                       // Block size (256 threads)
    ) {
        __shared__ double total_sum_shared[256];   // 2048 bytes
        __shared__ double entropy_shared[256];     // 2048 bytes

        // Each thread accumulates its portion of nodes
        double local_sum = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[i];
            if (val > 0.0) local_sum += val;
        }

        total_sum_shared[tid] = local_sum;
        __syncthreads();

        // Parallel reduction to sum all values
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                total_sum_shared[tid] += total_sum_shared[tid + s];
            }
            __syncthreads();
        }

        double total_sum = total_sum_shared[0];
        if (total_sum <= 0.0) return 0.0;

        // Each thread computes entropy for its portion
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

        // Parallel reduction for entropy
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                entropy_shared[tid] += entropy_shared[tid + s];
            }
            __syncthreads();
        }

        return entropy_shared[0] / (double)(n_sat);
    }

    // Main simulation kernel - one block processes one gene
    // Shared memory per block: 16 bytes (convergence tracking)
    __global__ void sepal_simulation(
        double* __restrict__ concentration_all,  // [n_genes, n_cells]
        double* __restrict__ derivatives_all,    // [n_genes, n_cells]
        const int* __restrict__ sat_idx,         // [n_sat, max_neighs]
        const int* __restrict__ unsat_idx,       // [n_unsat]
        double* __restrict__ results,            // [n_genes]
        int n_cells,
        int n_genes,
        int n_sat,
        int n_unsat,
        int max_neighs,
        int n_iter,
        double dt,
        double thresh,
        bool debug
    ) {
        int gene_idx = blockIdx.x;              // Each block handles one gene
        int tid = threadIdx.x;                  // Thread ID (0-255)
        int blockSize = blockDim.x;             // 256 threads per block

        if (gene_idx >= n_genes) return;

        // Per-gene pointers into global arrays
        double* concentration = &concentration_all[gene_idx * n_cells];
        double* derivatives = &derivatives_all[gene_idx * n_cells];

        // Convergence tracking (16 bytes shared per block)
        __shared__ double prev_entropy;         // 8 bytes
        __shared__ int convergence_iter;        // 4 bytes
        __shared__ bool converged_flag;         // 4 bytes (padded)

        if (tid == 0) {
            prev_entropy = 1.0;
            convergence_iter = -1;
            converged_flag = false;
        }
        __syncthreads();

        // Main iteration loop - all threads process their portion of nodes
        for (int iter = 0; iter < n_iter; iter++) {
            // Phase 1: Update derivatives for saturated nodes
            for (int i = tid; i < n_sat; i += blockSize) {
                double neighbor_sum = 0.0;
                for (int j = 0; j < max_neighs; j++) {
                    neighbor_sum += concentration[sat_idx[i * max_neighs + j]];
                }

                double center = concentration[i];
                double h = 1.0;
                double d2 = 0.0;

                if (max_neighs == 4) {
                    d2 = (neighbor_sum - 4.0 * center) / (h * h);
                } else if (max_neighs == 6) {
                    d2 = (neighbor_sum - 6.0 * center) / (h * h) * (2.0 / 3.0);
                }
                derivatives[i] = d2;
            }
            __syncthreads();

            // Phase 2: Update saturated node concentrations
            for (int i = tid; i < n_sat; i += blockSize) {
                concentration[i] += derivatives[i] * dt;
                concentration[i] = fmax(0.0, concentration[i]);
            }
            __syncthreads();

            // Phase 3: Update unsaturated nodes based on nearest saturated
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_global_idx = n_sat + i;
                concentration[unsat_global_idx] += derivatives[unsat_idx[i]] * dt;
                concentration[unsat_global_idx] = fmax(0.0, concentration[unsat_global_idx]);
            }
            __syncthreads();

            // Check convergence using entropy
            if (!converged_flag) {
                double current_entropy = compute_entropy_cooperative(
                    concentration, n_sat, tid, blockSize
                );

                if (tid == 0) {
                    double entropy_diff = fabs(current_entropy - prev_entropy);
                    if (entropy_diff <= thresh && convergence_iter == -1) {
                        convergence_iter = iter;
                        converged_flag = true;
                    }
                    prev_entropy = current_entropy;
                }
                __syncthreads();
            }

            if (converged_flag) break;
        }

        // Store result for this gene
        if (tid == 0) {
            results[gene_idx] = convergence_iter >= 0 ?
                (double)convergence_iter : -1.0;
        }
    }
}
