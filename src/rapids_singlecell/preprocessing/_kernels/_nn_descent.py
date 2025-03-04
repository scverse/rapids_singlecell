from __future__ import annotations

import cupy as cp

kernel_code_cos = r"""
extern "C" __global__
void computeDistances_Cosine(const float* data,
                    float* out,
                    const unsigned int* pairs,
                    const long long int n_samples,
                    const long long int n_features,
                    const long long int n_neighbors)
{
    long long int i1 = blockDim.x * blockIdx.x + threadIdx.x;
    if(i1 >= n_samples){
        return;
    }

    float sum_i1 = 0.0f;
    for (long long int d = 0; d < n_features; d++) {
        sum_i1 += powf(data[i1 * n_features + d], 2);
    }
    for (long long int j = 0; j < n_neighbors; j++){
        long long int i2 = static_cast<long long>(pairs[i1 * n_neighbors + j]);
        float dist = 0.0f;

        float sum_i2 = 0.0f;
        for (long long int d = 0; d < n_features; d++) {
            dist += data[i1 * n_features + d] * data[i2 * n_features + d];
            sum_i2 += powf(data[i2 * n_features + d], 2);
        }
        out[i1 * n_neighbors + j] = 1-dist/ (sqrtf(sum_i1) * sqrtf(sum_i2));
    }

}
"""

calc_distance_kernel_cos = cp.RawKernel(
    code=kernel_code_cos,
    name="computeDistances_Cosine",
)

kernel_code = r"""
extern "C" __global__
void computeDistances(const float* data,
                    float* out,
                    const unsigned int* pairs,
                    const long long int n_samples,
                    const long long int n_features,
                    const long long int n_neighbors)
{
    long long int i1 = blockDim.x * blockIdx.x + threadIdx.x;
    if(i1 >= n_samples){
        return;
    }
    for (long long int j = 0; j < n_neighbors; j++){
        long long int i2 = static_cast<long long>(pairs[i1 * n_neighbors + j]);
        float dist = 0.0f;
        for (long long int d = 0; d < n_features; d++) {
            float diff = data[i1 * n_features + d] - data[i2 * n_features + d];
            dist += powf(diff, 2);
        }
        out[i1 * n_neighbors + j] = dist;
    }
}
"""

calc_distance_kernel = cp.RawKernel(
    code=kernel_code,
    name="computeDistances",
)

kernel_code_inner = r"""
extern "C" __global__
void computeDistances_inner(const float* data,
                    float* out,
                    const unsigned int* pairs,
                    const long long int n_samples,
                    const long long int n_features,
                    const long long int n_neighbors)
{
    long long int i1 = blockDim.x * blockIdx.x + threadIdx.x;
    if(i1 >= n_samples){
        return;
    }


    for (long long int j = 0; j < n_neighbors; j++){
        long long int i2 = static_cast<long long>(pairs[i1 * n_neighbors + j]);
        float dist = 0.0f;

        for (long long int d = 0; d < n_features; d++) {
            dist += data[i1 * n_features + d] * data[i2 * n_features + d];

        }
        out[i1 * n_neighbors + j] =  dist;
    }

}
"""

calc_distance_kernel_inner = cp.RawKernel(
    code=kernel_code_inner,
    name="computeDistances_inner",
)
