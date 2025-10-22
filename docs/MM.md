# Memory Management

In rapids-singlecell, efficient memory management is crucial for handling large-scale datasets. This is facilitated by the integration of the RAPIDS Memory Manager ({mod}`rmm`). {mod}`rmm` is automatically invoked upon importing `rapids-singlecell`, modifying the default allocator for cupy. Integrating {mod}`rmm` with `rapids-singlecell` slightly modifies the execution speed of {mod}`cupy`. This change typically results in a minimal performance trade-off. However, it's crucial to be aware that certain specific functions, like {func}`~.pp.harmony_integrate`, might experience a more significant impact on performance efficiency due to this integration. Users can overwrite the default behavior with {func}`rmm.reinitialize`.

## Quick start

Pick one mode based on your dataset and hardware:

- If your data fits in GPU VRAM: use the pool allocator for speed → see [Pool Allocator](#pool-allocator).
- If your data is larger than VRAM: use managed memory to spill to host RAM → see [Managed Memory](#managed-memory).

Why not both? Pool allocation and managed memory target different trade-offs. Pooling assumes you keep data in VRAM and optimizes allocation speed. Managed memory assumes you will exceed VRAM and optimizes for correctness by spilling to host RAM. Combining both can negate benefits and increase fragmentation, so choose one.

## Managed Memory

- Purpose: use datasets larger than GPU VRAM by spilling to host RAM.
- How it works: VRAM oversubscription; data migrates between GPU and host as needed.
- Trade-off: slower than fully-in-VRAM; slowdown grows with how much you spill.
- Good for: very large datasets that otherwise OOM; exploratory or batch runs where correctness matters more than peak speed.

```python
# Enable `managed_memory`
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(managed_memory=True, pool_allocator=False)
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Pool Allocator

- Purpose: speed up allocations and reduce fragmentation when data fits in VRAM.
- How it works: pre-allocates a pool; subsequent allocations come from the pool.
- Trade-off: keeps memory reserved; needs sufficient VRAM.
- Good for: allocation-heavy steps (e.g., neighbor graphs, harmony integration) and repeated runs.

```python
# Enable `pool_allocator`
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator
rmm.reinitialize(
    managed_memory=False,
    pool_allocator=True,
)
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Best Practices
To achieve optimal memory management in rapids-singlecell, consider the following guidelines:

* **Large-scale Data Analysis:** Utilize `managed_memory` for datasets exceeding your VRAM's capacity, keeping in mind the potential performance penalties.
* **Performance-Critical Operations:** Choose `pool_allocator` when speed is critical and sufficient VRAM is available.
* **Do not enable both together:** Pooling prefers staying in VRAM; managed memory prefers spilling when needed. Mixing them can lead to unexpected performance and memory fragmentation.

### Troubleshooting

- CUDA out-of-memory (OOM) while using the pool allocator → switch to [Managed Memory](#managed-memory) or reduce dataset size.
- Very slow runtime with managed memory → reduce oversubscription or switch back to [Pool Allocator](#pool-allocator) if VRAM allows.

## Further Reading
For a more in-depth understanding of rmm and its functionalities, refer to the [RAPIDS Memory Manager documentation](https://docs.rapids.ai/api/rmm/stable/python/).


## System requirements and limits

rapids-singlecell performs most computations on the GPU. Ensure your system has a CUDA-capable GPU with sufficient VRAM for your datasets.

- With an RTX 3090, analyzing around 200,000 cells is typically feasible.
- With an A100 80GB, analyses with 1,000,000+ cells are possible.

For larger datasets, use {mod}`~rmm` managed memory to oversubscribe GPU memory to host RAM (similar to SWAP). This may introduce a performance penalty but can still outperform CPU-only runs. See the Managed Memory section above for how to enable it.

Limit note: For GPU-backed {class}`~anndata.AnnData`, the upper limit is governed by the sparse matrix `.nnz` value of 2**31-1 (2,147,483,647). This is due to the maximum `indptr` size currently supported by {mod}`~cupy` for sparse matrices.
