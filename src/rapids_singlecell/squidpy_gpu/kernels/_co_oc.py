from __future__ import annotations

import cupy as cp

kernel_code = r"""
extern "C" __global__
void occur_count_kernel(const float* spatial,
                        const float* thresholds, const int* label_idx,
                        int* result, int n, int k, int l_val)
{
    // Each block corresponds to a unique point pair (i, j).

    int i = blockIdx.x;  // grid is 1D over n*n
    int s = i%2;
    // Skip self-comparisons or out-of-bound indices.
    if (i >= n)
        return;
    int offset = (i % 4 < 2) ? 0 : l_val;
    float spx = spatial[i*2];
    float spy = spatial[i*2+1];
    int low = label_idx[i];
    for(int j = i+1; j< n; j++){
        float dx = spx - spatial[j*2];
        float dy = spy - spatial[j*2+1];
        float dist_sq = dx * dx + dy * dy;

        // Get labels for both points.
        int high = label_idx[j];
        if (high < low) {
            low, high = high, low;
        }
        if(s==0){
            low, high = high, low;
        }
        for (int r = threadIdx.x; r < l_val; r+=blockDim.x) {
            if (dist_sq <= thresholds[r]) {
                // Compute the flat index corresponding to result[label_i, label_j, r]
                //int index = label_i * (k * l_val) + label_j * l_val + r;
                int index = low * (k * l_val*2) + high * l_val*2 + r+offset;
                atomicAdd(&result[index], 1);
            }
        }
    }
}
"""
occur_count_kernel = cp.RawKernel(kernel_code, "occur_count_kernel")

kernel_code2 = r"""
extern "C" __global__
void occur_reduction_kernel(const int* result, float *out, int k, int l_val)
{
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index

    // Shared memory:
    //  - First k*k floats: will hold the counts matrix (converted to float).
    //  - Next k floats: used to store column sums.
    extern __shared__ float shared[];
    float* Y = shared;
    float* col_sum = shared + (k * k);

    int total_elements = k * k;

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x){
        Y[i] = 0.0f;
    }
    __syncthreads();

    // --- Load counts for this threshold and convert to float---
    for (int i = threadIdx.x; i < k; i += blockDim.x){
        for (int j = 0; j<k;j++){
            Y[i*k+j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
            Y[j*k+i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
            Y[i*k+j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
            Y[j*k+i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
        }
    }
    __syncthreads();

    // --- Compute total sum of the counts ---
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < k*k; idx += blockDim.x) {
        sum_val += Y[idx];
    }

    // Use warp-level reduction; since blockDim is 32, all threads are in one warp.
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }

    // Now, thread 0 in the warp holds the total.
    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // --- Normalize the matrix Y = Y / total (if total > 0) ---
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    }
    else{
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // --- Compute column sums of the normalized matrix ---
    // For each column j, sum over rows i: col_sum[j] = sum_{i=0}^{k-1} Y[i*k + j]
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // --- Compute conditional probabilities ---
    // For each row i, compute the row sum (over columns) then for each column j:
    //   cond = (row sum != 0) ? (Y[i*k+j] / row_sum) : 0.0f
    //   final = (col_sum[j] != 0) ? (cond / col_sum[j]) : 0.0f
    // We then store the result into out with layout (k, k, l_val) in C-order.
    // The correct linear index for element (i, j, r_th) is: i*(k*l_val) + j*l_val + r_th.
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }
        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f)
                cond = Y[i * k + j] / row_sum;
            float final_val = 0.0f;
            if (col_sum[j] != 0.0f)
                final_val = cond / col_sum[j];
            // Write to output: note the ordering (row, column, threshold)
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
}
"""
occur_count_kernel2 = cp.RawKernel(kernel_code2, "occur_reduction_kernel")
