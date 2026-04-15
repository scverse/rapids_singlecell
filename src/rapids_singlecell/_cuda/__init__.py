"""
CUDA extensions for rapids-singlecell (built via scikit-build-core/nanobind).

These modules provide GPU-accelerated kernels for various single-cell analysis
operations. Each module is compiled from CUDA source files and exposed through
nanobind bindings.

On systems without compiled extensions (e.g., docs builds), imports resolve
to None so that module-level imports don't raise ImportError.
"""

from __future__ import annotations

import importlib

# Pre-load librmm.so + deps so the dynamic linker can resolve them when
# our nanobind extensions (which link rmm) are imported.  This is the same
# pattern used by cuml, cuvs, and other RAPIDS packages.
try:
    import librmm

    librmm.load_library()
except (ImportError, OSError):
    pass

__all__ = [
    "_aggr_cuda",
    "_aucell_cuda",
    "_autocorr_cuda",
    "_bbknn_cuda",
    "_cooc_cuda",
    "_edistance_cuda",
    "_harmony_clustering_cuda",
    "_harmony_colsum_cuda",
    "_harmony_correction_batched_cuda",
    "_harmony_correction_cuda",
    "_harmony_kmeans_cuda",
    "_harmony_normalize_cuda",
    "_harmony_outer_cuda",
    "_harmony_pen_cuda",
    "_harmony_scatter_cuda",
    "_hvg_cuda",
    "_kde_cuda",
    "_ligrec_cuda",
    "_mean_var_cuda",
    "_nanmean_cuda",
    "_nn_descent_cuda",
    "_norm_cuda",
    "_pr_cuda",
    "_pv_cuda",
    "_qc_cuda",
    "_qc_dask_cuda",
    "_scale_cuda",
    "_sparse2dense_cuda",
    "_spca_cuda",
    "_wilcoxon_binned_cuda",
    "_wilcoxon_ovo_cuda",
    "_wilcoxon_ovr_cuda",
]


def __getattr__(name: str):
    if name in __all__:
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
