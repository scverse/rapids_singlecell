# scanpy-GPU

These functions offer accelerated near drop-in replacements for common tools porvided by {mod}`scanpy`.

## Preprocessing `pp`
Filtering of highly-variable genes, batch-effect correction, per-cell normalization.

Any transformation of the data matrix that is not a tool. Other than `tools`, preprocessing steps usually don’t return an easily interpretable annotation, but perform a basic transformation on the data matrix.

### Basic Preprocessing
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
### Batch effect correction

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.harmony_integrate
```

### Doublet detection
```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.scrublet
   pp.scrublet_simulate_doublets
```


### Neighbors
```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.neighbors
```

## Tools: `tl`

`tools` offers tools for the accelerated processing of {class}`~anndata.AnnData`. For visualization use {mod}`scanpy.pl`.

```{eval-rst}
.. module:: rapids_singlecell.tl
```

```{eval-rst}
.. currentmodule:: rapids_singlecell
```

### Embedding
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

### Clustering

```{eval-rst}
.. autosummary::
   :toctree: generated/

    tl.louvain
    tl.leiden
```

### Marker genes

```{eval-rst}
.. autosummary::
   :toctree: generated/

    tl.rank_genes_groups_logreg
```

## Plotting

For plotting please use scanpy's plotting API {mod}`scanpy.pl`.
