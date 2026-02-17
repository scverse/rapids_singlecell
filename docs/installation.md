# Installation
## Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* files provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder.
These *yaml* files install everything needed to run the example notebooks and get you started.

`````{tab-set}
````{tab-item} CUDA 13
```bash
conda env create -f conda/rsc_rapids_26.02_cuda13.yml
# or
mamba env create -f conda/rsc_rapids_26.02_cuda13.yml
```
*Python 3.13, CUDA 13.0*
````
````{tab-item} CUDA 12
```bash
conda env create -f conda/rsc_rapids_26.02_cuda12.yml
# or
mamba env create -f conda/rsc_rapids_26.02_cuda12.yml
```
*Python 3.13, CUDA 12.9*
````
`````

```{note}
RAPIDS currently doesn't support `channel_priority: strict`; use `channel_priority: flexible` instead
```

## PyPI

Starting with version 0.15.0, *rapids-singlecell* ships precompiled CUDA kernels via nanobind.
Prebuilt wheels are available for **x86_64** and **aarch64** Linux for both CUDA 12 and CUDA 13.

### Prebuilt wheels (recommended)

Install the wheel matching your CUDA version:

`````{tab-set}
````{tab-item} CUDA 13
```bash
pip install rapids-singlecell-cu13
```
````
````{tab-item} CUDA 12
```bash
pip install rapids-singlecell-cu12
```
````
`````

This installs the precompiled CUDA kernels but **not** the RAPIDS stack (cupy, cuml, cudf, etc.).
This is the recommended approach for **conda/mamba users** who already have RAPIDS installed in their environment.

### Prebuilt wheels with RAPIDS dependencies

To also install the RAPIDS stack via pip, use the `rapids` extra.
This requires the `--extra-index-url` flag for the NVIDIA PyPI index:

`````{tab-set}
````{tab-item} CUDA 13
```bash
pip install 'rapids-singlecell-cu13[rapids]' --extra-index-url=https://pypi.nvidia.com
```
````
````{tab-item} CUDA 12
```bash
pip install 'rapids-singlecell-cu12[rapids]' --extra-index-url=https://pypi.nvidia.com
```
````
`````

### Source distribution (self-compile)

The `rapids-singlecell` package on PyPI contains the source distribution.
Building from source requires a CUDA toolkit and a C++ compiler:

```bash
pip install rapids-singlecell
```

The CUDA kernels will be compiled during installation for your local GPU architecture.
You can select RAPIDS dependencies with the `rapids-cu12` or `rapids-cu13` extras:

```bash
pip install 'rapids-singlecell[rapids-cu12]' --extra-index-url=https://pypi.nvidia.com
```

```{note}
Building from source requires the CUDA toolkit (nvcc) and CMake >= 3.24 to be available in your environment.
```

## Docker

We also offer a Docker container for `rapids-singlecell`. This container includes all the necessary dependencies, making it even easier to get started with `rapids-singlecell`.

To use the Docker container, first, ensure that you have Docker installed on your system and that Docker supports the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
Then, you can pull our Docker image using the following command:

```
docker pull ghcr.io/scverse/rapids_singlecell:latest
```

To run the Docker container, use the following command:

```
docker run --rm --gpus all ghcr.io/scverse/rapids_singlecell:latest
```

The docker containers also work with apptainer (or singularity) on an HPC system.

First pull the container and wrap it in a `.sif` file:
```
apptainer pull rsc.sif ghcr.io/scverse/rapids_singlecell:latest
```
Then run the following command to execute the container:
```
apptainer run --nv rsc.sif
```


# System requirements

Most computations run on the GPU.
See the Memory Management page for hardware guidance, managed memory, and known limits:

- {doc}`memory_management`
