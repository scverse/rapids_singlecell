[![Stars](https://img.shields.io/github/stars/scverse/rapids_singlecell?logo=GitHub&color=blue)](https://github.com/scverse/rapids_singlecell/stargazers)
[![PyPI](https://img.shields.io/pypi/v/rapids-singlecell?logo=PyPI)](https://pypi.org/project/rapids-singlecell)
[![Downloads](https://static.pepy.tech/badge/rapids-singlecell)](https://pepy.tech/project/rapids-singlecell)
[![Documentation Status](https://readthedocs.org/projects/rapids-singlecell/badge/?version=latest)](https://rapids-singlecell.readthedocs.io/en/latest/?badge=latest)
[![Build and Test](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml/badge.svg)](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml)
[![Chat](https://img.shields.io/badge/zulip-join_chat-%2367b08f.svg)](https://scverse.zulipchat.com)

# rapids-singlecell: GPU-Accelerated Single-Cell Analysis within scverse

Rapids-singlecell offers enhanced single-cell data analysis as a near drop-in replacement predominantly for scanpy, while also incorporating select functionalities from squidpy and decoupler. Utilizing GPU computing with cupy and Nvidia’s RAPIDS, it emphasizes high computational efficiency. As part of the scverse ecosystem, rapids-singlecell continuously aims to maintain compatibility, adapting and growing through community collaboration.

* **Broad GPU Optimization:** Facilitates accelerated processing of large datasets, with GPU-enabled AnnData objects.
* **Selective scverse Library Integration:** Incorporates key functionalities from scanpy, with additional features from squidpy and decoupler.
* **Easy Installation Process:** Available via Conda and PyPI, with detailed setup guidelines.
* **Accessible Documentation:** Provides comprehensive guides and examples tailored for efficient application.

Our commitment with rapids-singlecell is to deliver a powerful, user-centric tool that significantly enhances single-cell data analysis capabilities in bioinformatics.

## Installation
### Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* file provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder. These *yaml* files install everything needed to run the example notbooks and get you started.
```
conda env create -f conda/rsc_rapids_24.02.yml
# or
mamba env create -f conda/rsc_rapids_23.12.yml
```
### PyPI
```
pip install rapids-singlecell
```
The default installer doesn't cover RAPIDS nor cupy. Information on how to install RAPIDS & cupy can be found [here](https://rapids.ai/start.html).

If you want to use RAPIDS PyPI packages, the whole library with all dependencies can be install with:
````
pip install 'rapids-singlecell[rapids11]' --extra-index-url=https://pypi.nvidia.com #CUDA11.X
pip install 'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com #CUDA12
````
It is important to ensure that the CUDA environment is set up correctly so that RAPIDS and Cupy can locate the necessary libraries.

## Documentation

Please have a look through the [documentation](https://rapids-singlecell.readthedocs.io/en/latest/)


## Citation

If you use this code, please cite: [![DOI](https://zenodo.org/badge/364573913.svg)](https://zenodo.org/badge/latestdoi/364573913)

Please also consider citing: [rapids-single-cell-examples](https://zenodo.org/badge/latestdoi/265649968) and  [scanpy](https://doi.org/10.1186/s13059-017-1382-0)

In addition to that please cite the methods' original research articles in the [scanpy documentation](https://scanpy.readthedocs.io/en/latest/references.html)

If you use the accelerated decoupler functions please cite [decoupler](https://doi.org/10.1093/bioadv/vbac016)
