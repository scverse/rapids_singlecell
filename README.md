[![Stars](https://img.shields.io/github/stars/scverse/rapids_singlecell?logo=GitHub&color=blue)](https://github.com/scverse/rapids_singlecell/stargazers)
[![PyPI](https://img.shields.io/pypi/v/rapids-singlecell?logo=PyPI)](https://pypi.org/project/rapids-singlecell)
[![Downloads](https://static.pepy.tech/badge/rapids-singlecell)](https://pepy.tech/project/rapids-singlecell)
[![Documentation Status](https://readthedocs.org/projects/rapids-singlecell/badge/?version=latest)](https://rapids-singlecell.readthedocs.io/en/latest/?badge=latest)
[![Build and Test](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml/badge.svg)](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml)

# rapids-singlecell

## Background
This library is designed to accelerate single cell data analysis by utilizing the capabilities of GPU computing. Drawing inspiration from both the [scanpy](https://github.com/scverse/scanpy) library by Theis lab and the [rapids-single-cell-examples](https://github.com/clara-parabricks/rapids-single-cell-examples) library from Nvidia's RAPIDS team, it introduces GPU-optimized versions of their functions. While aiming to remain compatible with the original codes, the library's primary objective is to blend the computational strength of GPUs with the user-friendly nature of the scverse ecosystem.

## Installation
### Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* file provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder. These *yaml* files install everything needed to run the example notbooks and get you started.
```
conda env create -f conda/rsc_rapids_23.04.yml
# or
mamba env create -f conda/rsc_rapids_23.08.yml
```
### PyPI
As of version 0.4.0 *rapids-singlecell* is now on PyPI.
```
pip install rapids-singlecell
```
The default installer doesn't cover RAPIDS nor cupy. Information on how to install RAPIDS & cupy can be found [here](https://rapids.ai/start.html).

If you want to use RAPIDS new PyPI packages, the whole library with all dependencies can be install with:
````
pip install 'rapids-singlecell[rapids11]' --extra-index-url=https://pypi.nvidia.com #CUDA11.X
pip install 'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com #CUDA12
````
It is important to ensure that the CUDA environment is set up correctly so that RAPIDS and Cupy can locate the necessary libraries.

To view a full guide how to set up a fully functioned single cell GPU accelerated conda environment visit [GPU_SingleCell_Setup](https://github.com/Intron7/GPU_SingleCell_Setup)

## Documentation

Please have a look through the [documentation](https://rapids-singlecell.readthedocs.io/en/latest/)


## Citation

If you use this code, please cite: [![DOI](https://zenodo.org/badge/364573913.svg)](https://zenodo.org/badge/latestdoi/364573913)

Please also consider citing: [rapids-single-cell-examples](https://zenodo.org/badge/latestdoi/265649968) and  [scanpy](https://doi.org/10.1186/s13059-017-1382-0)

In addition to that please cite the methods' original research articles in the [scanpy documentation](https://scanpy.readthedocs.io/en/latest/references.html)

If you use the accelerated decoupler functions please cite [decoupler](https://doi.org/10.1093/bioadv/vbac016)
