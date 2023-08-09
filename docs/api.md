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

## cunnData

```{eval-rst}
.. module:: rapids_singlecell.cunnData
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    cunnData.cunnData
```


## cunnData_funcs: `pp`

`cunnData_funcs` offers functions for preprocessing of {class}`~rapids_singlecell.cunnData.cunnData`.

### Preprocessing
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
### Plotting: `pl`

Basic plotting function for {class}`~rapids_singlecell.cunnData.cunnData` to assess the quality of the dataset.

```{eval-rst}
.. module:: rapids_singlecell.pl
.. currentmodule:: rapids_singlecell
.. autosummary::
   :toctree: generated/

   pl.scatter
   pl.violin

```

## scanpy-GPU: `tl`

`scanpy-GPU` offers tools for the accelerated processing of {class}`~anndata.AnnData`. For visualization use {mod}`scanpy.pl`.

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

    tl.pca
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

### Batch effect correction

```{eval-rst}
.. autosummary::
   :toctree: generated/

    tl.harmony_integrate
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
