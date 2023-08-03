[![Stars](https://img.shields.io/github/stars/scverse/rapids_singlecell?logo=GitHub&color=blue)](https://github.com/scverse/rapids_singlecell/stargazers)
[![PyPI](https://img.shields.io/pypi/v/rapids-singlecell?logo=PyPI)](https://pypi.org/project/rapids-singlecell)
[![PyPIDownloads](https://pepy.tech/badge/rapids-singlecell)](https://pepy.tech/project/rapids-singlecell)
[![Documentation Status](https://readthedocs.org/projects/rapids-singlecell/badge/?version=latest)](https://rapids-singlecell.readthedocs.io/en/latest/?badge=latest)

# rapids-singlecell

## Background
This repository offers some tools to make analyses of single cell datasets faster by running them on the GPU.
The functions are analogous versions of functions that can be found within [scanpy](https://github.com/scverse/scanpy) from the Theis lab or functions from [rapids-single-cell-examples](https://github.com/clara-parabricks/rapids-single-cell-examples) created by the Nvidia RAPIDS team. Most functions are kept close to the original code to ensure compatibility. My aim with this repository was to use the speedup that GPU computing offers and combine it with the ease of use from scanpy.

## News

I'm very honored to announce that I was invited to co-author a technical blog post that demonstrates the capabilities and performance of *rapids-singlecell* for NVIDIA. You can read through the blog [here](https://developer.nvidia.com/blog/gpu-accelerated-single-cell-rna-analysis-with-rapids-singlecell/?ncid=so-link-660513-vt12&=&linkId=100000207171999#cid=an01_so-link_en-us).\
As always, your thoughts and feedback are valued, as they contribute to the ongoing refinement and development of *rapids-singlecell*.


## Installation
### Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* file provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder. These *yaml* files install everything needed to run the example notbooks and get you started.
```
conda env create -f conda/rsc_rapids_23.04.yml
# or
mamba env create -f conda/rsc_rapids_23.06.yml
```
### PyPI
As of version 0.4.0 *rapids-singlecell* is now on PyPI.
```
pip install rapids-singlecell
```
The default installer doesn't cover RAPIDS nor cupy. Information on how to install RAPIDS & cupy can be found [here](https://rapids.ai/start.html).

If you want to use RAPIDS new PyPI packages, the whole library with all dependencies can be install with:
````
pip install 'rapids-singlecell[rapids]' --extra-index-url=https://pypi.nvidia.com
````
Please note that the RAPIDS PyPI packages are still considered experimental. It is important to ensure that the CUDA environment is set up correctly so that RAPIDS and Cupy can locate the necessary libraries.

To view a full guide how to set up a fully functioned single cell GPU accelerated conda environment visit [GPU_SingleCell_Setup](https://github.com/Intron7/GPU_SingleCell_Setup)

## Documentation

Please have a look through the [documentation](https://rapids-singlecell.readthedocs.io/en/latest/)


## Citation

If you use this code, please cite: [![DOI](https://zenodo.org/badge/364573913.svg)](https://zenodo.org/badge/latestdoi/364573913)

Please also consider citing: [rapids-single-cell-examples](https://zenodo.org/badge/latestdoi/265649968) and  [scanpy](https://doi.org/10.1186/s13059-017-1382-0)

In addition to that please cite the methods' original research articles in the [scanpy documentation](https://scanpy.readthedocs.io/en/latest/references.html)

If you use the accelerated decoupler functions please cite [decoupler](https://doi.org/10.1093/bioadv/vbac016)

## Notebooks
To show the capability of these functions, I created two example notebooks evaluating the same workflow running on the CPU and GPU. These notebooks should run in the environment, that is described in Requirements. First, run the `data_downloader` notebook to create the AnnData object for the analysis. If you run both `demo_cpu` and `demo_gpu` you should see a big speedup when running the analyses on the GPU.
