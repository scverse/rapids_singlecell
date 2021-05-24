# rapids_singlecell

## Background
This repository offers some tools to make the analysis of single cells datasets faster by running them on the GPU. 
The functions are analogous versions of functions that can be found within [scanpy](https://github.com/theislab/scanpy) from the Theis lab or functions from [rapids-single-cell-examples](https://github.com/clara-parabricks/rapids-single-cell-examples) created by the Nvidia Rapids team. Most functions are kept close to the original code to ensure compatibility. My aim with this repository was to use the speedup that GPU computing offers and combine it with the ease of use from scanpy.

## Requirements

To run the code in this repository you need a conda environment with rapids and scanpy installed. To use the full functionality of this repo please use `rapids-0.20a`, because this version come with cupy 9. You also need an Nvidia GPU.
```
conda create -n rapids-0.20_sc -c rapidsai-nightly -c nvidia -c conda-forge -c bioconda \
    rapids-blazing=0.20 python=3.8 cudatoolkit=11.2 cudnn cutensor cusparselt scanpy \
    leidenalg louvain multicore-tsne python-wget
conda activate rapids-0.20_sc
python -m ipykernel install --user --display-name "rapids-0.20_sc"
```

With this enviroment, you should be able to run the notebooks. So far I have only tested these Notebooks on a Quadro RTX 6000 and an RTX 3090.

## Functions

### cunnData
The preprocessing of the single-cell data is performed with the `cunnData`. It is a replacement for the [AnnData](https://github.com/theislab/anndata) object used by scanpy. The `cunnData` object is a cutdown version of an `AnnData` object. At its core lies a sparse matrix (`.X`) within the GPU memory. `.obs` and `.var` are pandas data frame and `.uns` is a dictionary. Most preprocessing functions of `scanpy` are methods of the `cunnData` class. I tried to keep the input as close to the original scanpy implementation as possible.
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
* Diffusion Maps
* rank_genes_groups with logistic regression
* some plotting functions for cunnData objects

## Notebooks
To show the capability of these functions, I created two example notebooks to show the same workflow running on the CPU and GPU. These notebooks should run in the environment, that is described in Requirements. First, run the `data_downloader` notebook to create the AnnData object for the analysis. If you run both `demo_gpu` and `demo_gpu` you should see a big speedup when running the analysis on the GPU.
