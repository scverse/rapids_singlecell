# rapids-singlecell

## Background
This repository offers some tools to make analyses of single cell datasets faster by running them on the GPU. 
The functions are analogous versions of functions that can be found within [scanpy](https://github.com/scverse/scanpy) from the Theis lab or functions from [rapids-single-cell-examples](https://github.com/clara-parabricks/rapids-single-cell-examples) created by the Nvidia RAPIDS team. Most functions are kept close to the original code to ensure compatibility. My aim with this repository was to use the speedup that GPU computing offers and combine it with the ease of use from scanpy.

## Installation
### Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* file provided in the [conda](https://github.com/Intron7/rapids_singlecell/tree/main/conda) folder. These *yaml* files install everything needed to run the example notbooks and get you started.
```
conda create -f conda/rsc_rapids_22.12.yml
# or
mamba create -f conda/rsc_rapids_23.02a.yml
```
### PyPI
As of version 0.4.0 *rapids-singlecell* is now on PyPI.
```
pip install rapids-singlecell
```
The default installer doesn't cover RAPIDS nor cupy. Information on how to install RAPIDS & cupy can be found [here](https://rapids.ai/start.html).

If you want to use RAPIDS new PyPI packages, the whole library with all dependencies can be install with:
````
pip install 'rapids-singlecell[rapids]’ --extra-index-url=https://pypi.ngc.nvidia.com
````
Please note that the RAPIDS PyPI packages are still considered experimental. It is important to ensure that the CUDA environment is set up correctly so that RAPIDS and Cupy can locate the necessary libraries.

To view a full guide how to set up a fully functioned single cell GPU accelerated conda environment visit [GPU_SingleCell_Setup](https://github.com/Intron7/GPU_SingleCell_Setup)

## Citation

If you use this code, please cite: [![DOI](https://zenodo.org/badge/364573913.svg)](https://zenodo.org/badge/latestdoi/364573913)

Please also consider citing: [rapids-single-cell-examples](https://zenodo.org/badge/latestdoi/265649968) and  [scanpy](https://doi.org/10.1186/s13059-017-1382-0)

In addition to that please cite the methods' original research articles in the [scanpy documentation](https://scanpy.readthedocs.io/en/latest/references.html)

If you use the accelerated decoupler functions please cite [decoupler](https://doi.org/10.1093/bioadv/vbac016)

## Functionality

### cunnData
The preprocessing of the single-cell data is performed with *cunnData*. It is a replacement for the [AnnData](https://github.com/scverse/anndata) object used by scanpy. The *cunnData* object is a cutdown version of an *AnnData* object. At its core lies a sparse matrix (`.X`) within the GPU memory. `.obs` and `.var` are pandas data frame and `.uns` is a dictionary. It also supports `.layers`, `.varm` and `.obsm`. `.layers` are stored on the GPU, while `.obsm` and `.varm` are not.
Since version `0.3.0` you can use cunnData for spatial transcriptomics datasets.\
*cunnData* includes methods for:
* `__getiem__` to filter the object based on `.obs` and `.var`.
* `__repr__` 
* transform cunnData object to AnnData object

### cunnData_funcs or pp
Most preprocessing functions of *scanpy* are reimplemented for the *cunnData* class. I tried to keep the input as close to the original scanpy implementation as possible.
Please have look at the notebooks to assess the functionality. I tried to write informative docstrings for each function.\
*cunnData_funcs* includes functions for:
* filter genes based on cells expressing that genes
* filter cells based on a multitude of parameters (eg. number of expressed genes, mitchondrial content)
* caluclate_qc (based on scanpy's `pp.calculate_qc_metrics`)
* normalize_total
* normalize based on *pearson_residuals*
* log1p
* highly_variable_genes
  * seurat
  * cellranger
  * seurat_v3
  * pearson_residuals
  * poisson_gene_selection (adapted from *scvi*)
* regress_out
* scale
* PCA (PCA/ incremental PCA/ truncated svd)
* some plotting functions of qc parameters


### scanpy_gpu or tl
*scanpy_gpu* are functions that are written to directly work with an *AnnData* object and replace the scanpy counterpart by running on the GPU. Scanpy already supports GPU versions of `pp.neighbors` and `tl.umap` using RAPIDS.\
*scanpy_gpu* includes additional functions for:
* PCA (PCA/ incremental PCA/ truncated svd)
* Leiden Clustering
* Louvain Clustering
* TSNE
* Kmeans Clustering 
* Kernel Density
* Harmony Integration (gpu port of [harmonypy](https://github.com/slowkow/harmonypy))
* Diffusion Maps
* PyMDE (adapted from *scvi*)
* Force Atlas 2 (draw_grah) 
* rank_genes_groups with logistic regression

### decoupler_gpu
[Decoupler](https://github.com/saezlab/decoupler-py/) is an amazing toolkit, that contains different statistical methods to extract biological activities from omics data within a unified framework. So far I have reimplemented `run_mlm` and `run_wsum` to run on the GPU. As always I tried to keep the syntax as close the original as possible. *decoupler_gpu* also works with the same `models` as decoupler. For a closer looks please check out the `demo_gpu.ipynb` in `notebooks`.\
*decoupler_gpu* includes additional functions for:
* run_mlm
* run_wsum

## Notebooks
To show the capability of these functions, I created two example notebooks evaluating the same workflow running on the CPU and GPU. These notebooks should run in the environment, that is described in Requirements. First, run the `data_downloader` notebook to create the AnnData object for the analysis. If you run both `demo_cpu` and `demo_gpu` you should see a big speedup when running the analyses on the GPU.

## Benchmarks

Here are some benchmarks. I ran the notebook on the CPU with as many cores as were available where possible. 

|Step                          |CPU (Ryzen 5950x, 32 Cores, 64GB RAM)|GPU (RTX 3090)|CPU (AMD Eypc Rome, 30 Cores, 500GB RAM)| GPU (Quadro RTX 6000)|GPU (A100 80GB)|
|------------------------------|---------------------------|--------------|----------|--------------|----------------|
|whole Notebook                | 728 s                     | 43 s         | 917 s    | 67 s         | 57 s           |
|Preprocessing                 | 75 s                      | 21 s         | 40 s     | 34 s         | 30 s           |
|Clustering and Visulatization | 423 s                     | 18 s         | 524 s    | 27 s         | 21 s           |
|Normalize_total               | 252 ms                    | > 1ms        | 425 ms   | 1 ms         | 1 ms           |
|Highly Variable Genes         | 3.2 s                     | 2.6 s        | 4.1 s    | 2.7 s        | 3.7 s          |
|Regress_out                   | 63 s                      | 2 s          | 24 s     | 2 s          | 2 s            |
|Scale                         | 1.3 s                     | 299 ms       | 2 s      | 2  s         | 359 ms         |
|PCA                           | 26 s                      | 1.8 s        | 23 s     | 3.6 s        | 2.6 s          |
|Neighbors                     | 10 s                      | 5 s          | 16.8 s   | 8.1  s       | 6 s            |
|UMAP                          | 30 s                      | 659 ms       | 66 s     | 1 s          | 783 ms         |
|Louvain                       | 16 s                      | 121 ms       | 20 s     | 214 ms       | 201 ms         |
|Leiden                        | 11 s                      | 102 ms       | 20 s     | 175 ms       | 152 ms         |
|TSNE                          | 240 s                     | 1.4 s        | 319 s    | 1.8 s        | 1.4 s          |
|Logistic_Regression           | 74 s                      | 4 s          | 45 s     | 5 s          | 3.4 s          |
|Diffusion Map                 | 715 ms                    | 259 ms       | 747 ms   | 431 ms       | 826 ms         |
|Force Atlas 2                 | 207 s                     | 236 ms       | 300 s    | 298 ms       | 353 ms         |

I also observed that the first GPU run in a new enviroment is slower than the runs after that (with a restarted kernel) (RTX 6000).
