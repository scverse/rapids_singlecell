from __future__ import annotations

import cupy as cp

_find_top_k_kernel = r"""
extern "C" __global__
void find_top_k_per_row(
    const float* __restrict__ data,
    const int* __restrict__ indptr,
    const int n_rows,
    const int trim,
    float* __restrict__ vals) {

    extern __shared__ float shared_memory[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];
    int length = end - start;

    if (length <= trim) {
        vals[row] = 0.0f;  // Or another default value indicating insufficient elements
        return;
    }

    // Each thread has its own top_k array in shared memory
    int thread_idx = threadIdx.x;
    int shared_offset = thread_idx * trim;
    float* top_k = &shared_memory[shared_offset];

    // Initialize top_k with 0
    for (int i = 0; i < trim; ++i) {
        top_k[i] = 0;
    }

    // Process each element in the row
    int min_index = 0;
    for (int idx = start; idx < end; ++idx) {
        if (data[idx] <= top_k[min_index]) continue;

        // If current value is larger than the smallest in top_k, replace it
        top_k[min_index] = data[idx];

        // Find the new smallest value in top_k and set min_index
        for (int i = 0; i < trim; ++i) {
            if (top_k[i] < top_k[min_index]) {
                min_index = i;
            }
        }
    }

    // After processing, use min_index to write the smallest value in top_k to vals
    float kth_largest = top_k[min_index];
    vals[row] = kth_largest;
}
"""

# Compile the kernel
find_top_k_per_row_kernel = cp.RawKernel(_find_top_k_kernel, "find_top_k_per_row")

_cut_smaller_kernel = r"""
extern "C" __global__
void cut_smaller(
    const int *indptr,
    const int * index,
    float *data,
    float* vals,
    int n_rows) {
    int row_id = blockIdx.x;
    if(row_id >= n_rows){
        return;
    }
    int start_idx = indptr[row_id];
    int stop_idx = indptr[row_id+1];

    float cut_row = vals[row_id];
    for(int i = start_idx+threadIdx.x; i < stop_idx; i+= blockDim.x){
        float cut = max(vals[index[i]], cut_row);
        if(data[i]<cut){
            data[i] = 0;
        }

    }

}
"""
cut_smaller_func = cp.RawKernel(_cut_smaller_kernel, "cut_smaller")
