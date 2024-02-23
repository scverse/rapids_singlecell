# Memory Management

In rapids-singlecell, efficient memory management is crucial for handling large-scale datasets. This is facilitated by the integration of the RAPIDS Memory Manager ({mod}`rmm`). {mod}`rmm` is automatically invoked upon importing `rapids-singlecell`, modifying the default allocator for cupy. Integrating {mod}`rmm` with `rapids-singlecell` slightly modifies the execution speed of {mod}`cupy`. This change typically results in a minimal performance trade-off. However, it's crucial to be aware that certain specific functions, like {func}`~rapids_singlecell.pp.harmony`, might experience a more significant impact on performance efficiency due to this integration. Users can overwrite the default behavior with {func}`rmm.reinitialize`.

## Managed Memory

In {mod}`rmm`, the `managed_memory` feature facilitates VRAM oversubscription, allowing for the processing of data structures larger than the default VRAM capacity. This effectively extends the memory limit up to twice the VRAM size. Leveraging managed memory will introduce a performance overhead. This is particularly evident with substantial oversubscription, as it necessitates increased dependency on the comparatively slower system memory, leading to slowdowns in data processing tasks.

```
# Enable `managed_memory`
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
rmm.reinitialize(
    managed_memory=True,
    pool_allocator=False,
)
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Pool Allocator

The `pool_allocator` functionality in {mod}`rmm` optimizes memory handling by pre-allocating a pool of memory, which can be swiftly accessed for GPU-related tasks. This approach, while being more memory-intensive, significantly boosts performance. It is particularly beneficial for operations that are heavy on memory usage, such as {func}`~rapids_singlecell.pp.harmony`, by minimizing the time spent on dynamic memory allocation during runtime.

```
# Enable `pool_allocator`
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
rmm.reinitialize(
    managed_memory=False,
    pool_allocator=True,
)
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Best Practices
To achieve optimal memory management in rapids-singlecell, consider the following guidelines:

* Large-scale Data Analysis: Utilize managed_memory for datasets exceeding your VRAM's capacity, keeping in mind the potential performance penalties.
* Performance-Critical Operations: Choose pool_allocator when speed is critical and sufficient VRAM is available.

## Further Reading
For a more in-depth understanding of rmm and its functionalities, refer to the [RAPIDS Memory Manager documentation](https://docs.rapids.ai/api/rmm/stable/python/).
