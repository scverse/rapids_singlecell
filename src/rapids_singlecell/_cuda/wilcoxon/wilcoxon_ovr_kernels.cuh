#pragma once

/** Count nonzeros per column from CSR. One thread per row. */
__global__ void csr_col_histogram_kernel(const int* __restrict__ indices,
                                         const int* __restrict__ indptr,
                                         int* __restrict__ col_counts,
                                         int n_rows, int n_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int rs = indptr[row];
    int re = indptr[row + 1];
    for (int p = rs; p < re; ++p) {
        int c = indices[p];
        if (c < n_cols) atomicAdd(&col_counts[c], 1);
    }
}

/**
 * Scatter CSR nonzeros into CSC layout for columns [col_start, col_stop).
 * write_pos[c - col_start] must be initialized to the prefix-sum offset
 * for column c.  Each thread atomically claims a unique destination slot.
 */
__global__ void csr_scatter_to_csc_kernel(
    const float* __restrict__ data, const int* __restrict__ indices,
    const int* __restrict__ indptr, int* __restrict__ write_pos,
    float* __restrict__ csc_vals, int* __restrict__ csc_row_idx, int n_rows,
    int col_start, int col_stop) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int rs = indptr[row];
    int re = indptr[row + 1];
    // Binary search for col_start (overflow-safe midpoint)
    int lo = rs, hi = re;
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (indices[m] < col_start)
            lo = m + 1;
        else
            hi = m;
    }
    for (int p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        int dest = atomicAdd(&write_pos[c - col_start], 1);
        csc_vals[dest] = data[p];
        csc_row_idx[dest] = row;
    }
}

/**
 * Decide whether to use shared or global memory for OVR rank accumulators.
 * Returns the smem size to request and sets use_gmem accordingly.
 */
static int query_max_smem_per_block() {
    static int cached = -1;
    if (cached < 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMaxSharedMemoryPerBlock,
                               device);
    }
    return cached;
}

static size_t ovr_smem_config(int n_groups, bool& use_gmem) {
    size_t need = (size_t)(n_groups + 32) * sizeof(double);
    if ((int)need <= query_max_smem_per_block()) {
        use_gmem = false;
        return need;
    }
    // Fall back to global memory accumulators; only need warp buf in smem
    use_gmem = true;
    return 32 * sizeof(double);
}

/**
 * Decide smem-vs-gmem for the sparse OVR rank kernel.  Two accumulator
 * arrays (grp_sums + grp_nz_count) of size n_groups each plus warp buf.
 */
static size_t sparse_ovr_smem_config(int n_groups, bool& use_gmem) {
    size_t need = (size_t)(2 * n_groups + 32) * sizeof(double);
    if ((int)need <= query_max_smem_per_block()) {
        use_gmem = false;
        return need;
    }
    use_gmem = true;
    return 32 * sizeof(double);
}

/**
 * Fill sort values with row indices [0,1,...,n_rows-1] per column.
 * Grid: (n_cols,), block: 256 threads.
 */
__global__ void fill_row_indices_kernel(int* __restrict__ vals, int n_rows,
                                        int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    int* out = vals + (long long)col * n_rows;
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        out[i] = i;
    }
}
