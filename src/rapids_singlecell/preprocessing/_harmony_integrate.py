from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def harmony_integrate(
    adata: AnnData,
    key: str,
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    dtype: type = np.float64,
    **kwargs,
) -> None:
    """
    Use harmonypy to integrate different experiments.
    Harmony is an algorithm for integrating single-cell
    data from multiple experiments. This function uses the python
    gpu-computing based port of Harmony, to integrate single-cell data
    stored in an AnnData object. As Harmony works by adjusting the
    principal components, this function should be run after performing
    PCA but before computing the neighbor graph.

    Parameters
    ----------
        adata
            The annotated data matrix.
        key
            The name of the column in ``adata.obs`` that differentiates among experiments/batches.
        basis
            The name of the field in ``adata.obsm`` where the PCA table is
            stored. Defaults to ``'X_pca'``, which is the default for
            ``sc.tl.pca()``.
        adjusted_basis
            The name of the field in ``adata.obsm`` where the adjusted PCA
            table will be stored after running this function. Defaults to
            ``X_pca_harmony``.
        dtype
            The data type to use for the Harmony. If you use 32-bit you may experience
            numerical instability.
        kwargs
            Any additional arguments will be passed to
            ``harmonpy_gpu.run_harmony()``.

    Returns
    -------
        Updates adata with the field ``adata.obsm[adjusted_basis]``, \
        containing principal components adjusted by Harmony such that \
        different experiments are integrated.

    """
    from . import _harmonypy_gpu

    X = adata.obsm[basis].astype(dtype)

    harmony_out = _harmonypy_gpu.run_harmony(X, adata.obs, key, dtype=dtype, **kwargs)

    adata.obsm[adjusted_basis] = harmony_out.Z_corr.T.get()
