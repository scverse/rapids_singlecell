# Usage Principles

## Import

```
import rapids_singlecell as rsc
```

## Workflow

The workflow of *rapids-singlecell* is basically the same as *scanpy's*. The main difference is the speed at which *rsc* can analyze the data. For more information please checkout the notebooks and API documentation.

### AnnData setup

With the release of version 0.10.0 {class}`~anndata.AnnData` supports GPU arrays and Sparse Matrices.

Rapids-singlecell leverages this capability to perform analyses directly on GPU-enabled {class}`~anndata.AnnData` objects. This also leads to the depreciation of {class}`~rapids_singlecell.cunnData.cunnData` and it's removal in early 2024.

To get your {class}`~anndata.AnnData` object onto the GPU you can set {attr}`~anndata.AnnData.X` or each {attr}`~anndata.AnnData.layers` to a GPU based matrix.

```
adata.X = cpx.scipy.sparse.csr_matrix(adata.X)  # moves `.X` to the GPU
adata.X = adata.X.get() # moves `.X` back to the GPU
```

You can also use the {mod}`rapids_singlecell.utils` to move arrays and matrices.

```
rsc.utils.anndata_to_GPU(adata) # moves `.X` to the GPU
rsc.utils.anndata_to_CPU(adata) # moves `.X` to the CPU
```

### Preprocessing

The preprocessing can  be handled by {class}`~anndata.AnnData` and {class}`~rapids_singlecell.cunnData.cunnData`. It offers accelerated versions of functions within {mod}`scanpy.pp`.

Example:
```
rsc.pp.highly_variable_genes(adata,n_top_genes=5000,flavor="seurat_v3",batch_key= "PatientNumber",layer = "counts")
adata = adata[:,adata.var["highly_variable"]==True]
rsc.pp.regress_out(adata,keys=["n_counts", "percent_MT"])
rsc.pp.scale(adata,max_value=10)
```
After preprocessing is done just transform the {class}`~rapids_singlecell.cunnData.cunnData` into {class}`~anndata.AnnData` and continue the analysis.

### Tools

The functions provided in {mod}`~.tl` are designed to as near drop-in replacements for the functions in {mod}`scanpy.tl`, but offer significantly improved performance. Consequently, you can continue to use scanpy's plotting API.

Example:
```
rsc.tl.tsne(adata)
sc.pl.tsne(adata, color="leiden")
```

### Decoupler-GPU

`dcg` offers accelerated drop in replacements for {func}`~rapids_singlecell.dcg.run_mlm` and {func}`~rapids_singlecell.dcg.run_wsum`

Example:
```
import decoupler as dc
model = dc.get_progeny(organism='human', top=100)
rsc.dcg.run_mlm(mat=adata, net=net, source='source', target='target', weight='weight', verbose=True)
acts_mlm = dc.get_acts(adata, obsm_key='mlm_estimate')
sc.pl.umap(acts_mlm, color=['KLF5',"FOXA1", 'CellType'], cmap='coolwarm', vcenter=0)
```

### cunnData (depreciated)

```{image} _static/cunndata.svg
:width: 500px
```

The {class}`~rapids_singlecell.cunnData.cunnData` object can replace {class}`~anndata.AnnData` for preprocessing. All {mod}`~.pp` functions (except {func}`~.pp.neighbors`) are aimed towards cunnData. {attr}`~.X` and {attr}`~.layers` are stored on the GPU. The other components are stored in the host memory.
