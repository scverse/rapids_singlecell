// kernels/edistance_kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

/**
 * Compute between-group mean distances for off-diagonal pairs only
 * Each block processes one group pair, threads collaborate within the block
 */
__global__ void compute_group_distances(
    const float* __restrict__ embedding,
    const int* __restrict__ cat_offsets,
    const int* __restrict__ cell_indices,
    const int* __restrict__ pair_left,
    const int* __restrict__ pair_right,
    float* __restrict__ d_other,
    int k,
    int n_features)
{
    extern __shared__ float shared_sums[];

    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    float local_sum = 0.0f;

    const int a = pair_left[block_id];
    const int b = pair_right[block_id];

    // No need to check a == b since we only pass off-diagonal pairs

    const int start_a = cat_offsets[a];
    const int end_a = cat_offsets[a + 1];
    const int start_b = cat_offsets[b];
    const int end_b = cat_offsets[b + 1];

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    // Compute between-group distances (ALL cross-pairs)
    for (int ia = start_a + thread_id; ia < end_a; ia += block_size) {
        const int idx_i = cell_indices[ia];

        for (int jb = start_b; jb < end_b; ++jb) {
            const int idx_j = cell_indices[jb];

            float dist_sq = 0.0f;
            #pragma unroll
            for (int feat = 0; feat < n_features; ++feat) {
                float diff = embedding[idx_i * n_features + feat] -
                            embedding[idx_j * n_features + feat];
                dist_sq += diff * diff;
            }
            local_sum += sqrtf(dist_sq);
        }
    }

    // Reduce across threads using shared memory
    shared_sums[thread_id] = local_sum;
    __syncthreads();
    
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_sums[thread_id] += shared_sums[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        // Store mean between-group distance
        d_other[block_id] = shared_sums[0] / (float)(n_a * n_b);
    }
}

} // extern "C"