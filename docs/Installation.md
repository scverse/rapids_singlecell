# Installation
## Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* file provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder. These *yaml* files install everything needed to run the example notebooks and get you started.
```
conda env create -f conda/rsc_rapids_24.06.yml
# or
mamba env create -f conda/rsc_rapids_24.04.yml
```
## PyPI
As of version 0.4.0 *rapids-singlecell* is now on PyPI.
```
pip install rapids-singlecell
```
The default installer doesn't cover RAPIDS nor cupy. Information on how to install RAPIDS & cupy can be found [here](https://rapids.ai/start.html).

If you want to use RAPIDS new PyPI packages, the whole library with all dependencies can be install with:
```
pip install 'rapids-singlecell[rapids11]' --extra-index-url=https://pypi.nvidia.com #CUDA11.X
pip install 'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com #CUDA12

```
It is important to ensure that the CUDA environment is set up correctly so that RAPIDS and Cupy can locate the necessary libraries.

To view a full guide how to set up a fully functioned single cell GPU accelerated conda environment visit [GPU_SingleCell_Setup](https://github.com/Intron7/GPU_SingleCell_Setup)


# GPU-Memory and System Requirements

*rapids-singlecell* relays for most computation on the GPU. A GPU with sufficient VRAM is therefore required to handle large datasets.
With a RTX 3090 it's possible to analyze 200000 cells without any issues. With an A100 80GB it is even possible to analyze more than 1000000. For even larger datasets, {mod}`~rmm` is required to oversubscribe GPU memory into host memory, similar to SWAP memory. However, using `managed_memory` can result in a performance penalty, but this is still preferable to CPU runtimes.

The upper limit for GPU-based {class}`~anndata.AnnData` is a `.nnz` (non-zero elements) value of 2**31-1 (2147483647). This constraint is due to the maximum `indptr` (index pointer array for compressed sparse format) size that {mod}`~cupy` currently supports for sparse matrices.
