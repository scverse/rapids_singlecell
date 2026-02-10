from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_mul_kernel_csr = r"""
(const int *indptr, {0} *data, int nrows, int tsum) {
    int row = blockIdx.x;
    if(row >= nrows)
        return;

    int start = indptr[row];
    int end = indptr[row + 1];

    // Parallel row sum
    {0} val = 0.0;
    for(int i = start + threadIdx.x; i < end; i += blockDim.x)
        val += data[i];

    // Warp-level reduction via shuffle
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ {0} warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if(laneIdx == 0)
        warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : ({0})0.0;
    if(warpIdx == 0) {
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast scale to all threads
    if(threadIdx.x == 0)
        warp_sums[0] = (val > 0.0) ? (({0})tsum / val) : ({0})0.0;
    __syncthreads();

    // Parallel multiply
    {0} scale = warp_sums[0];
    if(scale > 0.0) {
        for(int i = start + threadIdx.x; i < end; i += blockDim.x)
            data[i] *= scale;
    }
}
"""

_mul_kernel_dense = r"""
({0} *data, int nrows, int ncols, int tsum) {
    int row = blockIdx.x;
    if(row >= nrows)
        return;

    int row_offset = row * ncols;

    // Parallel row sum
    {0} val = 0.0;
    for(int i = threadIdx.x; i < ncols; i += blockDim.x)
        val += data[row_offset + i];

    // Warp-level reduction via shuffle
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ {0} warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if(laneIdx == 0)
        warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : ({0})0.0;
    if(warpIdx == 0) {
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast scale to all threads
    if(threadIdx.x == 0)
        warp_sums[0] = (val > 0.0) ? (({0})tsum / val) : ({0})0.0;
    __syncthreads();

    // Parallel multiply
    {0} scale = warp_sums[0];
    if(scale > 0.0) {
        for(int i = threadIdx.x; i < ncols; i += blockDim.x)
            data[row_offset + i] *= scale;
    }
}
"""

_get_sparse_sum_major_kernel = r"""
(const int *indptr, const {0} *data, {0} *sums, int major) {
    int major_idx = blockIdx.x;
    if(major_idx >= major)
        return;

    int start_idx = indptr[major_idx];
    int stop_idx = indptr[major_idx + 1];

    {0} val = 0.0;
    for(int i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x)
        val += data[i];

    // Warp-level reduction via shuffle
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ {0} warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if(laneIdx == 0)
        warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : ({0})0.0;
    if(warpIdx == 0) {
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if(threadIdx.x == 0)
        sums[major_idx] = val;
}
"""


def _mul_csr(dtype):
    return cuda_kernel_factory(_mul_kernel_csr, (dtype,), "_mul_kernel_csr")


def _mul_dense(dtype):
    return cuda_kernel_factory(_mul_kernel_dense, (dtype,), "_mul_kernel_dense")


def _get_sparse_sum_major(dtype):
    return cuda_kernel_factory(
        _get_sparse_sum_major_kernel, (dtype,), "_get_sparse_sum_major_kernel"
    )


_prescaled_mul_kernel_csr = r"""
(const int *indptr, {0} *data, const {0} *scales, int nrows) {
    int row = blockIdx.x;
    if(row >= nrows)
        return;

    {0} scale = scales[row];
    int start = indptr[row];
    int end = indptr[row + 1];

    for(int i = start + threadIdx.x; i < end; i += blockDim.x)
        data[i] *= scale;
}
"""

_prescaled_mul_kernel_dense = r"""
({0} *data, const {0} *scales, int nrows, int ncols) {
    int row = blockIdx.x;
    if(row >= nrows)
        return;

    {0} scale = scales[row];
    int row_offset = row * ncols;
    for(int i = threadIdx.x; i < ncols; i += blockDim.x)
        data[row_offset + i] *= scale;
}
"""

_find_hi_genes_kernel_csr = r"""
(const int *indptr, const int *indices, const {0} *data,
 bool *gene_is_hi, {0} max_fraction, int nrows) {
    int row = blockIdx.x;
    if(row >= nrows)
        return;

    int start = indptr[row];
    int end = indptr[row + 1];

    // Phase 1: accumulate partial row sum in register
    {0} val = 0.0;
    for(int i = start + threadIdx.x; i < end; i += blockDim.x)
        val += data[i];

    // Warp-level reduction via shuffle
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction (supports up to 256 threads / 8 warps)
    int nWarps = blockDim.x >> 5;
    __shared__ {0} warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if(laneIdx == 0)
        warp_sums[warpIdx] = val;
    __syncthreads();

    // First warp reduces all warp sums
    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : ({0})0.0;
    if(warpIdx == 0) {
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast threshold to all threads
    if(threadIdx.x == 0)
        warp_sums[0] = max_fraction * val;
    __syncthreads();

    // Phase 2: check elements against threshold
    {0} threshold = warp_sums[0];
    for(int i = start + threadIdx.x; i < end; i += blockDim.x) {
        if(data[i] > threshold)
            gene_is_hi[indices[i]] = true;
    }
}
"""

_masked_sum_major_kernel = r"""
(const int *indptr, const int *indices, const {0} *data,
 const bool *gene_mask, {0} *sums, int major) {
    int major_idx = blockIdx.x;
    if(major_idx >= major)
        return;

    int start_idx = indptr[major_idx];
    int stop_idx = indptr[major_idx + 1];

    // Accumulate partial sum in register, skipping masked genes
    {0} val = 0.0;
    for(int i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
        if(!gene_mask[indices[i]])
            val += data[i];
    }

    // Warp-level reduction via shuffle
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction (supports up to 256 threads / 8 warps)
    int nWarps = blockDim.x >> 5;
    __shared__ {0} warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if(laneIdx == 0)
        warp_sums[warpIdx] = val;
    __syncthreads();

    // First warp reduces all warp sums
    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : ({0})0.0;
    if(warpIdx == 0) {
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if(threadIdx.x == 0)
        sums[major_idx] = val;
}
"""


def _prescaled_mul_csr(dtype):
    return cuda_kernel_factory(
        _prescaled_mul_kernel_csr, (dtype,), "_prescaled_mul_kernel_csr"
    )


def _prescaled_mul_dense(dtype):
    return cuda_kernel_factory(
        _prescaled_mul_kernel_dense, (dtype,), "_prescaled_mul_kernel_dense"
    )


def _find_hi_genes_csr(dtype):
    return cuda_kernel_factory(
        _find_hi_genes_kernel_csr, (dtype,), "_find_hi_genes_kernel_csr"
    )


def _masked_sum_major(dtype):
    return cuda_kernel_factory(
        _masked_sum_major_kernel, (dtype,), "_masked_sum_major_kernel"
    )


_masked_mul_kernel_csr = r"""
(const int *indptr, const int *indices, {0} *data,
 const bool *gene_mask, int nrows, {0} tsum) {
    int row = blockIdx.x;
    if(row >= nrows)
        return;

    int start = indptr[row];
    int end = indptr[row + 1];

    // Parallel masked row sum
    {0} val = 0.0;
    for(int i = start + threadIdx.x; i < end; i += blockDim.x) {
        if(!gene_mask[indices[i]])
            val += data[i];
    }

    // Warp-level reduction via shuffle
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ {0} warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if(laneIdx == 0)
        warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : ({0})0.0;
    if(warpIdx == 0) {
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast scale to all threads
    if(threadIdx.x == 0)
        warp_sums[0] = (val > 0.0) ? (tsum / val) : ({0})0.0;
    __syncthreads();

    // Parallel multiply
    {0} scale = warp_sums[0];
    if(scale > 0.0) {
        for(int i = start + threadIdx.x; i < end; i += blockDim.x)
            data[i] *= scale;
    }
}
"""


def _masked_mul_csr(dtype):
    return cuda_kernel_factory(
        _masked_mul_kernel_csr, (dtype,), "_masked_mul_kernel_csr"
    )
