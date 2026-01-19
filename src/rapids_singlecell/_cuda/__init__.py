from __future__ import annotations

"""
CUDA extensions for rapids-singlecell (built via scikit-build-core/nanobind).

These modules provide GPU-accelerated kernels for various single-cell analysis
operations. Each module is compiled from CUDA source files and exposed through
nanobind bindings.
"""

__all__ = [
    # Aggregation
    "_aggr_cuda",
    # AUCell scoring
    "_aucell_cuda",
    # Spatial autocorrelation (Moran's I, Geary's C)
    "_autocorr_cuda",
    # BBKNN trimming
    "_bbknn_cuda",
    # Co-occurrence analysis
    "_cooc_cuda",
    # Harmony integration
    "_harmony_colsum_cuda",
    "_harmony_kmeans_cuda",
    "_harmony_normalize_cuda",
    "_harmony_outer_cuda",
    "_harmony_pen_cuda",
    "_harmony_scatter_cuda",
    # Ligand-receptor analysis
    "_ligrec_cuda",
    # Mean-variance computation
    "_mean_var_cuda",
    # NaN-aware mean
    "_nanmean_cuda",
    # NN-descent distance calculation
    "_nn_descent_cuda",
    # Normalization
    "_norm_cuda",
    # Pearson residuals / HVG
    "_pr_cuda",
    # Permutation testing (decoupler)
    "_pv_cuda",
    # QC metrics
    "_qc_cuda",
    "_qc_dask_cuda",
    # Scaling
    "_scale_cuda",
    # Sparse to dense conversion
    "_sparse2dense_cuda",
    # Sparse PCA
    "_spca_cuda",
]
