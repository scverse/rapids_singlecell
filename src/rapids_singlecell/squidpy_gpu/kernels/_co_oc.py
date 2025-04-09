from __future__ import annotations

import cupy as cp

kernel_code_pairwise = r"""
extern "C" __global__
void occur_count_kernel_pairwise(const float* __restrict__ spatial,
                        const float* __restrict__ thresholds,
                        const int* __restrict__ label_idx,
                        int* __restrict__ result,
                        int n,
                        int k,
                        int l_val)
{
    int i = blockIdx.x;  // grid is 1D over n*n
    int s = i % 2;
    if (i >= n)
        return;
    int offset = (i % 4 < 2) ? 0 : l_val;
    float spx = spatial[i * 2];
    float spy = spatial[i * 2 + 1];
    int label_i = label_idx[i];

    for (int j = i + 1; j < n; j++) {
        float dx = spx - spatial[j * 2];
        float dy = spy - spatial[j * 2 + 1];
        float dist_sq = dx * dx + dy * dy;

        // Get labels for both points
        int low = label_i;
        int high = label_idx[j];

        // Sort labels if needed
        if (high < low) {
            int tmp = low;
            low = high;
            high = tmp;
        }

        // Swap based on s flag
        if (s == 0) {
            int tmp = low;
            low = high;
            high = tmp;
        }

        // Process each threshold in parallel within the block
        for (int r = threadIdx.x; r < l_val; r += blockDim.x) {
            if (dist_sq <= thresholds[r]) {
                int index = low * (k * l_val * 2) + high * l_val * 2 + r + offset;
                atomicAdd(&result[index], 1);
            }
        }
    }
}
"""
occur_count_kernel_pairwise = cp.RawKernel(
    kernel_code_pairwise, "occur_count_kernel_pairwise"
)

kernel_code_pairwise_fast = r"""
extern "C" __global__
void occur_count_kernel_pairwise_fast(const float* __restrict__ spatial,
                            const float* __restrict__ thresholds,
                            const int* __restrict__ label_idx,
                            int* __restrict__ result,
                            int n,
                            int k,
                            int l_val)
{
    extern __shared__ float shared[];
    float* Y = shared;
    int i = blockIdx.x;
    int r = blockIdx.y;
    for (int j = threadIdx.x; j < k * blockDim.x ; j+= blockDim.x){
        Y[j] = 0;
    }
    __syncthreads();

    float spx = spatial[i * 2];
    int low = label_idx[i];
    float spy = spatial[i * 2 + 1];
    for(int j = i+1+ threadIdx.x; j< n; j+= blockDim.x){
        float dx = spx - spatial[j * 2];
        float dy = spy - spatial[j * 2 + 1];
        float dist_sq = dx * dx + dy * dy;
        int high = label_idx[j];
        if (dist_sq <= thresholds[r]) {
            int index = k * threadIdx.x + high;
            Y[index] += 1;
        }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < k; j+= blockDim.x){
        float sum = 0;
        for (int t = 0; t < blockDim.x; t++){
            int index = k * t + j;
            sum += Y[index];
        }
        if (low < j){
            if (sum>0) atomicAdd(&result[r*(k*k)+low*k+j], sum);
        }
        else{
            if (sum>0) atomicAdd(&result[r*(k*k)+j*k+low], sum);
        }
    }
    __syncthreads();
}
"""
occur_count_kernel_pairwise_fast = cp.RawKernel(
    kernel_code_pairwise_fast, "occur_count_kernel_pairwise_fast"
)


occur_reduction_kernel_code_shared = r"""
extern "C" __global__
void occur_reduction_kernel_shared(const int* __restrict__ result,
                            float* __restrict__ out,
                            int k,
                            int l_val,
                            int format)
{
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index

    // Shared memory allocation
    extern __shared__ float shared[];
    float* Y = shared;
    float* col_sum = shared + (k * k);

    int total_elements = k * k;

    // Initialize shared memory
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        Y[i] = 0.0f;
    }
    __syncthreads();

    // --- Load counts for this threshold and convert to float---
    if (format == 0){
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
            }
        }
    }
    else{
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                Y[i * k + j] += float(result[r_th * (k * k) + i * k + j]);
                Y[j * k + i] += float(result[r_th * (k * k) + i * k + j]);
            }
        }
    }
    __syncthreads();

    // Compute total sum of the counts
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        sum_val += Y[idx];
    }

    // Warp-level reduction
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }

    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // Normalize the matrix Y = Y / total (if total > 0)
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // Compute column sums of the normalized matrix
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // Compute conditional probabilities
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }

        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f) {
                cond = Y[i * k + j] / row_sum;
            }

            float final_val = 0.0f;
            if (col_sum[j] != 0.0f) {
                final_val = cond / col_sum[j];
            }

            // Write to output with (row, column, threshold) ordering
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
    __syncthreads();
}
"""
occur_reduction_kernel_shared = cp.RawKernel(
    occur_reduction_kernel_code_shared, "occur_reduction_kernel_shared"
)

occur_reduction_kernel_code_global = r"""
extern "C" __global__
void occur_reduction_kernel_global(const int* __restrict__ result,
                            float* __restrict__ inter_out,
                            float* __restrict__ out,
                            int k,
                            int l_val,
                            int format)
{
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index
    if (r_th >= l_val)
        return;
    // Shared memory allocation
    extern __shared__ float shared[];
    float* Y = inter_out + r_th*k*k;
    float* col_sum = shared;

    int total_elements = k * k;

    // --- Load counts for this threshold and convert to float---
    if (format == 0){
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
            }
        }
    }
    else{
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                Y[i * k + j] += float(result[r_th * (k * k) + i * k + j]);
                Y[j * k + i] += float(result[r_th * (k * k) + i * k + j]);
            }
        }
    }
    __syncthreads();

    // Compute total sum of the counts
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        sum_val += Y[idx];
    }
    __syncthreads();
    // Warp-level reduction
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // Normalize the matrix Y = Y / total (if total > 0)
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // Compute column sums of the normalized matrix
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // Compute conditional probabilities
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }

        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f) {
                cond = Y[i * k + j] / row_sum;
            }

            float final_val = 0.0f;
            if (col_sum[j] != 0.0f) {
                final_val = cond / col_sum[j];
            }

            // Write to output with (row, column, threshold) ordering
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
    __syncthreads();
}
"""
occur_reduction_kernel_global = cp.RawKernel(
    occur_reduction_kernel_code_global, "occur_reduction_kernel_global"
)
