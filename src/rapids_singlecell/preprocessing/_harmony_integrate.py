from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
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
    correction_method: Literal["fast", "original"] = "original",
    use_gemm: bool|None = None,
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
        correction_method
            Choose which method for the correction step: ``original`` for original method, ``fast`` for improved method.
        use_gemm
            If True, use a One-Hot-Encoding Matrix and GEMM to compute Harmony. If False use a label vector. This is more memory efficient and faster for large datasets with a large number of batches. Defaults to True for more than 30 batches.
        kwargs
            Any additional arguments will be passed to
            ``harmonpy_gpu.run_harmony()``.

    Returns
    -------
        Updates adata with the field ``adata.obsm[adjusted_basis]``, \
        containing principal components adjusted by Harmony such that \
        different experiments are integrated.

    """
    from ._harmony import harmonize

    X = adata.obsm[basis].astype(dtype)
    if isinstance(X, np.ndarray):
        X = cp.array(X)
    harmony_out = harmonize(
        X, adata.obs.copy(), key, correction_method=correction_method, use_gemm=use_gemm, **kwargs
    )

    adata.obsm[adjusted_basis] = harmony_out.get()
