```{eval-rst}
.. module:: rapids_singlecell
```

```{eval-rst}
.. automodule:: rapids_singlecell
   :noindex:
```
# API

Import rapids-singlecell as:

```
import rapids_singlecell as rsc
```

## scanpy_GPU

These functions offer accelerated near drop-in replacements for common tools porvided by `scanpy`.

### Preprocessing `pp`
Filtering of highly-variable genes, batch-effect correction, per-cell normalization.

Any transformation of the data matrix that is not a tool. Other than `tools`, preprocessing steps usually don’t return an easily interpretable annotation, but perform a basic transformation on the data matrix.

All `preprocessing` functions work with {class}`~rapids_singlecell.cunnData.cunnData` except {func}`~rapids_singlecell.pp.neighbors`

#### Basic Preprocessing
```{eval-rst}
.. module:: rapids_singlecell.pp
.. currentmodule:: rapids_singlecell
.. autosummary::
   :toctree: generated/

   pp.calculate_qc_metrics
   pp.filter_cells
   pp.filter_genes
   pp.normalize_total
   pp.log1p
   pp.highly_variable_genes
   pp.regress_out
   pp.scale
   pp.pca
   pp.normalize_pearson_residuals
   pp.flag_gene_family
   pp.filter_highly_variable
```
#### Batch effect correction

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.harmony_integrate
```

#### Neighbors
```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.neighbors
```

### Tools: `tl`

`tools` offers tools for the accelerated processing of {class}`~anndata.AnnData`. For visualization use {mod}`scanpy.pl`.

```{eval-rst}
.. module:: rapids_singlecell.tl
```

```{eval-rst}
.. currentmodule:: rapids_singlecell
```

#### Embedding
```{eval-rst}
.. autosummary::
   :toctree: generated/

    tl.umap
    tl.tsne
    tl.diffmap
    tl.draw_graph
    tl.mde
    tl.embedding_density
```

#### Clustering

```{eval-rst}
.. autosummary::
   :toctree: generated/

    tl.louvain
    tl.leiden
```

#### Marker genes

```{eval-rst}
.. autosummary::
   :toctree: generated/

    tl.rank_genes_groups_logreg
```

### Plotting

For plotting please use scanpy's plotting API {mod}`scanpy.pl`.

### Utils

These functions offer convineant ways to move arrays and matrices from and to the GPU.
```{eval-rst}
.. module:: rapids_singlecell.utils
.. currentmodule:: rapids_singlecell
.. autosummary::
   :toctree: generated/
    utlis.anndata_to_GPU
    utlis.anndata_to_CPU
```



## squidpy-GPU: `gr`

{mod}`squidpy.gr` is a tool for the analysis of spatial molecular data. {mod}`rapids_singlecell.gr` acclerates some of these functions.

```{eval-rst}
.. module:: rapids_singlecell.gr
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    gr.spatial_autocorr
    gr.ligrec
```

## decoupler-GPU: `dcg`

{mod}`decoupler` contains different statistical methods to extract biological activities. {mod}`rapids_singlecell.dcg` acclerates some of these methods.

```{eval-rst}
.. module:: rapids_singlecell.dcg
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    dcg.run_mlm
    dcg.run_wsum
```

## cunnData

{class}`~rapids_singlecell.cunnData.cunnData` is depreciated and will be removed in 2024. Please start switching to {class}`~anndata.AnnData`

```{eval-rst}
.. module:: rapids_singlecell.cunnData
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    cunnData.cunnData
```
