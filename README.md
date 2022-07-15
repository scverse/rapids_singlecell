# rapids_singlecell

## Background
This repository offers some tools to make analyses of single cell datasets faster by running them on the GPU. 
The functions are analogous versions of functions that can be found within [scanpy](https://github.com/theislab/scanpy) from the Theis lab or functions from [rapids-single-cell-examples](https://github.com/clara-parabricks/rapids-single-cell-examples) created by the Nvidia Rapids team. Most functions are kept close to the original code to ensure compatibility. My aim with this repository was to use the speedup that GPU computing offers and combine it with the ease of use from scanpy.

## Requirements

To run the code in this repository you need a conda environment with rapids and scanpy installed. To use the full functionality of this repo please use `rapids-22.04` or `rapids-22.06`. You also need an Nvidia GPU.
```
conda create -n rapids_singelcell -f conda/rapids_singecell.yml
conda activate rapids_singelcell
ipython kernel install --user --name=rapids_singelcell
```
After you set up the enviroment you can install this package from this wheel into the enviroment. The wheel doesn't install any dependencies
```
pip install https://github.com/Intron7/rapids_singlecell/releases/download/v0.2.0/rapids_singlecell-0.2.0-py3-none-any.whl
```

With this enviroment, you should be able to run the notebooks. So far I have tested these Notebooks on an A100 80GB, a Quadro RTX 6000 and a RTX 3090.

## Citation

If you use this code, please cite: [![DOI](https://zenodo.org/badge/364573913.svg)](https://zenodo.org/badge/latestdoi/364573913)

Please also consider citing: [rapids-single-cell-examples](https://zenodo.org/badge/latestdoi/265649968) and  [scanpy](https://doi.org/10.1186/s13059-017-1382-0)

In addition to that please cite the methods' original research articles in the [scanpy documentation](https://scanpy.readthedocs.io/en/latest/references.html)

## Functions

### cunnData
The preprocessing of the single-cell data is performed with `cunnData`. It is a replacement for the [AnnData](https://github.com/theislab/anndata) object used by scanpy. The `cunnData` object is a cutdown version of an `AnnData` object. At its core lies a sparse matrix (`.X`) within the GPU memory. `.obs` and `.var` are pandas data frame and `.uns` is a dictionary. It also supports layers. Most preprocessing functions of `scanpy` are methods of the `cunnData` class. I tried to keep the input as close to the original scanpy implementation as possible.
Please have look at the notebooks to assess the functionality. I tried to write informative docstrings for each method. 
`cunnData` includes methods for:
* filter genes based on cells expressing that genes
* filter cells based on a multitude of parameters (eg. number of expressed genes, mitchondrial content)
* caluclate_qc (based on scanpy's `pp.calculate_qc_metrics`)
* normalize_total
* log1p
* highly_varible_genes
* regress_out 
* scale
* transform `cunnData` object to `AnnData` object

### scanpy_gpu_funcs
`scanpy_gpu_funcs` are functions that are written to directly work with an `AnnData` object and replace the scanpy counterpart by running on the GPU. Scanpy already supports GPU versions of `pp.neighbors` and `tl.umap` using rapids.
`scanpy_gpu_funcs` includes additional functions for:
* PCA
* Leiden Clustering
* Louvain Clustering
* TSNE
* Kmeans Clustering 
* Kernel Density
* Diffusion Maps
* Force Atlas 2 (draw_grah) 
* rank_genes_groups with logistic regression
* some plotting functions for cunnData objects

## Notebooks
To show the capability of these functions, I created two example notebooks evaluating the same workflow running on the CPU and GPU. These notebooks should run in the environment, that is described in Requirements. First, run the `data_downloader` notebook to create the AnnData object for the analysis. If you run both `demo_gpu` and `demo_gpu` you should see a big speedup when running the analyses on the GPU.

## Benchmarks

Here are some benchmarks. I ran the notebook on the CPU with as many cores as were available where possible. 

|Step                          |CPU (Ryzen 5950x, 32 Cores, 64GB RAM)|GPU (RTX 3090)|CPU (AMD Eypc Rome, 30 Cores, 500GB RAM)| GPU (Quadro RTX 6000)|GPU (A100 80GB)|
|------------------------------|---------------------------|--------------|----------|--------------|----------------|
|whole Notebook                | 728 s                     | 43 s         | 917 s    | 67 s         | 57 s           |
|Preprocessing                 | 75 s                      | 21 s         | 40 s     | 34 s         | 30 s           |
|Clustering and Visulatization | 423 s                     | 18 s         | 524 s    | 27 s         | 21 s           |
|Normalize_total               | 252 ms                    | > 1ms        | 425 ms   | 1 ms         | 1 ms           |
|Highly Variable Genes         | 3.2 s                     | 2.6 s        | 4.1 s    | 2.7 s        | 3.7 s          |
|Regress_out                   | 63 s                      | 14 s         | 24 s     | 23 s         | 15 s           |
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
