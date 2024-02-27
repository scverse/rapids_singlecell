# Usage Principles

## Import

```
import rapids_singlecell as rsc
```

## Workflow

The workflow of *rapids-singlecell* is basically the same as *scanpy's*. The main difference is the speed at which *rsc* can analyze the data. For more information please checkout the notebooks and API documentation.

### AnnData setup

{class}`~anndata.AnnData` supports GPU arrays and Sparse Matrices.

Rapids-singlecell leverages this capability to perform analyses directly on GPU-enabled {class}`~anndata.AnnData` objects.

To get your {class}`~anndata.AnnData` object onto the GPU you can set {attr}`~anndata.AnnData.X` or each {attr}`~anndata.AnnData.layers` to a GPU based matrix.

```
adata.X = cpx.scipy.sparse.csr_matrix(adata.X)  # moves `.X` to the GPU
adata.X = adata.X.get() # moves `.X` back to the CPU
```

You can also use {mod}`rapids_singlecell.get` to move arrays and matrices.

```
rsc.get.anndata_to_GPU(adata) # moves `.X` to the GPU
rsc.get.anndata_to_CPU(adata) # moves `.X` to the CPU
```

### Preprocessing

The preprocessing can be handled by the functions in {mod}`~.pp`. They offer accelerated versions of functions within {mod}`scanpy.pp`.

Example:
```
rsc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3", batch_key= "PatientNumber", layer = "counts")
adata = adata[:,adata.var["highly_variable"]==True]
rsc.pp.regress_out(adata,keys=["n_counts", "percent_MT"])
rsc.pp.scale(adata,max_value=10)
```

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
