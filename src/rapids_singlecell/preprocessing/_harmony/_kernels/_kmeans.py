from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_kmeans_err_kernel_code = r"""(const {0}* __restrict__ r,
                    const {0}* __restrict__ dot,
                    size_t n,
                    {0}* __restrict__ out)
{
    // --- per-thread accumulator -------------
    {0} acc = {0}(0);

    using Vec = {0}4;

    // grid-stride loop, vectorised load  -----
    size_t i = (blockIdx.x*blockDim.x + threadIdx.x) * 4;
    const size_t stride = gridDim.x*blockDim.x*4;

    while (i + 3 < n) {
        Vec r4   = *(const Vec*)(r   + i);
        Vec dot4 = *(const Vec*)(dot + i);

        acc += r4.x * {0}(2) * ({0}(1) - dot4.x);
        acc += r4.y * {0}(2) * ({0}(1) - dot4.y);
        acc += r4.z * {0}(2) * ({0}(1) - dot4.z);
        acc += r4.w * {0}(2) * ({0}(1) - dot4.w);
        i += stride;
    }
    // tail elements
    while (i < n) {
        {0} rv   = r[i];
        {0} dotv = dot[i];
        acc += rv * {0}(2) * ({0}(1) - dotv);
        i ++;
    }


    // --- warp-shuffle reduction -------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    // --- block reduce -----------------------
    static __shared__ {0} s[32];              // one per warp
    if ((threadIdx.x & 31) == 0) s[threadIdx.x>>5] = acc;
    __syncthreads();

    if (threadIdx.x < 32) {
        {0} val = (threadIdx.x < (blockDim.x>>5)) ? s[threadIdx.x] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) atomicAdd(out, val);
    }
}
"""


def _get_kmeans_err_kernel(dtype):
    return cuda_kernel_factory(
        _kmeans_err_kernel_code,
        (dtype,),
        "kmeans_err_kernel",
    )
