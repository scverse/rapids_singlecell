# Installation
## Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* file provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder. These *yaml* files install everything needed to run the example notebooks and get you started.
```
conda env create -f conda/rsc_rapids_25.08.yml #default CUDA-12.9 PYTHON-3.13
# or
mamba env create -f conda/rsc_rapids_25.10.yml #default CUDA-12.9 PYTHON-3.13
```
```{note}
RAPIDS currently doesn't support `channel_priority: strict`; use `channel_priority: flexible` instead
```

## PyPI
*rapids-singlecell* is also on PyPI.
```
pip install rapids-singlecell
```
The default installer doesn't cover RAPIDS nor CuPy. Information on how to install RAPIDS & CuPy can be found [here](https://rapids.ai/start.html).

If you want to use RAPIDS new PyPI packages, the whole library with all dependencies can be install with:
```
uv pip install 'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com #CUDA12
uv pip install 'rapids-singlecell[rapids13]' --extra-index-url=https://pypi.nvidia.com #CUDA13


```
It is important to ensure that the CUDA environment is set up correctly so that RAPIDS and CuPy can locate the necessary libraries.

```{note}
If you are using `python=3.12` with `uv`, you might need to add the `--index-strategy=unsafe-best-match` flag to ensure compatibility.
```

## Docker

We also offer a Docker container for `rapids-singlecell`. This container includes all the necessary dependencies, making it even easier to get started with `rapids-singlecell`.

To use the Docker container, first, ensure that you have Docker installed on your system and that Docker supports the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html). Then, you can pull our Docker image using the following command:

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

Most computations run on the GPU. See the Memory Management page for hardware guidance, managed memory, and known limits:

- {doc}`MM`
