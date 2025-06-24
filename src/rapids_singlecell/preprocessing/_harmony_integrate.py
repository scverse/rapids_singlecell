from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

    from ._harmony import COLSUM_ALGO


def harmony_integrate(
    adata: AnnData,
    key: str | list[str],
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    dtype: type = np.float64,
    correction_method: Literal["fast", "original"] = "original",
    use_gemm: bool = False,
    colsum_algo: COLSUM_ALGO | None = None,
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
            The key(s) of the column(s) in ``adata.obs`` that differentiates among experiments/batches.
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
            If True, use a One-Hot-Encoding Matrix and GEMM to compute Harmony. If False use a label vector. A label vector is more memory efficient and faster for large datasets with a large number of batches.
        colsum_algo
            Choose which algorithm to use for column sums. If `None`, choose the algorithm based on the number of rows and columns. If `'benchmark'`, benchmark all algorithms and choose the best one.
        kwargs
            Any additional arguments will be passed to
            ``_harmony.harmonize``.

    Returns
    -------
        Updates adata with the field ``adata.obsm[adjusted_basis]``, \
        containing principal components adjusted by Harmony such that \
        different experiments are integrated.

    """
    from ._harmony import harmonize

    # Ensure the basis exists in adata.obsm
    if basis not in adata.obsm:
        raise ValueError(
            f"The specified basis '{basis}' is not available in adata.obsm. "
            f"Available bases: {list(adata.obsm.keys())}"
        )

    # Get the input data
    input_data = adata.obsm[basis]

    try:
        # Handle different array types
        if isinstance(input_data, np.ndarray):
            # NumPy array: convert directly to CuPy
            try:
                X = cp.array(input_data, dtype=dtype, order="C")
            except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
                raise MemoryError(
                    "Not enough GPU memory to allocate array. "
                    "Try reducing the dataset size or using a GPU with more memory."
                ) from e
        elif isinstance(input_data, cp.ndarray):
            # CuPy array: ensure correct dtype and layout with a copy
            X = input_data.astype(dtype, order="C", copy=False)
        else:
            # Other array types: convert to NumPy first, then to CuPy
            try:
                # Try to convert to numpy (works for most array-like objects)
                np_array = np.array(input_data, dtype=dtype, order="C")
                X = cp.array(np_array, dtype=dtype, order="C")
            except Exception as e:
                raise TypeError(
                    f"Could not convert input of type {type(input_data).__name__} to CuPy array: {str(e)}"
                ) from e

        # Verify array is valid
        if cp.isnan(X).any():
            raise ValueError(
                "Input data contains NaN values. Please handle these before running harmony_integrate."
            )

    except Exception as e:
        raise RuntimeError(f"Error preparing data for Harmony: {str(e)}") from e

    harmony_out = harmonize(
        X,
        adata.obs,
        key,
        correction_method=correction_method,
        use_gemm=use_gemm,
        colsum_algo=colsum_algo,
        **kwargs,
    )

    adata.obsm[adjusted_basis] = harmony_out.get()
