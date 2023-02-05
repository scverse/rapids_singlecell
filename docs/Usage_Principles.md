# Usage Principles

## Import

```
import rapids_singlecell as rsc
```

## cunnData

```{image} _static/cunndata.svg
:width: 500px
```

The {class}`~rapids_singlecell.cunnData.cunnData` object replaces {class}`~anndata.AnnData` for preprocessing. All {mod}`~.pp` and {mod}`~.pl` functions are aimed towards cunnData. {attr}`~.X` and {attr}`~.layers` are stored on the GPU. The other components are stored in the host memory.

## Workflow

The workflow of *rapids-singlecell* is basically the same as *scanpy's*. The main difference is the speed at which *rsc* can analyze the data. For more information please checkout the notebooks and API documentation.

### Preprocessing

The preprocessing is handled by {class}`~rapids_singlecell.cunnData.cunnData` and `cunnData_funcs`. The latter is import as {mod}`~.pp` and {mod}`~.pl` to mimic the behavior of scanpy. 

Example:
```
rsc.pp.highly_variable_genes(cudata,n_top_genes=5000,flavor="seurat_v3",batch_key= "PatientNumber",layer = "counts")
cudata = cudata[:,cudata.var["highly_variable"]==True]
rsc.pp.regress_out(cudata,keys=["n_counts", "percent_MT"])
rsc.pp.scale(cudata,max_value=10)
```
After preprocessing is done just transform the {class}`~rapids_singlecell.cunnData.cunnData` into {class}`~anndata.AnnData` and continue the analysis.

### Tools

The functions in {mod}`~.tl` are created to manipulate the {class}`~anndata.AnnData` object. They are near dropin replacements for *scanpy's* functions just a lot faster. For this reason you can use *scanpy's* plotting API. All {mod}`~.tl` functions work on {class}`~anndata.AnnData`, for the reason {func}`~.harmony_integrate` has switched from `.pp` to `.tl`. This is also why there are two functions to calculate the princpal components one for {class}`~rapids_singlecell.cunnData.cunnData`[`.pp`] and one for {class}`~anndata.AnnData`[`.tl`]. 

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
