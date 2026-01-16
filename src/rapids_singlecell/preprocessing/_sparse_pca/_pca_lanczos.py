"""
PCA using Lanczos bidiagonalization SVD for sparse matrices.

This module provides a PCA implementation optimized for sparse matrices on GPU,
using Lanczos bidiagonalization with implicit restarts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import cupy as cp

from rapids_singlecell.preprocessing._utils import _get_mean_var

if TYPE_CHECKING:
    from cupyx.scipy.sparse import spmatrix

from ._operators import MeanCenteredOperator
from ._svd_lanczos import gpu_sparse_svds


class PCA_sparse_lanczos:
    """
    PCA for sparse matrices using Lanczos bidiagonalization SVD.

    This implementation is memory efficient for large sparse matrices as it
    avoids computing the full covariance matrix.

    Parameters
    ----------
    n_components
        Number of principal components to compute.
    zero_center
        If True, compute standard PCA (mean-centered).
        If False, compute truncated SVD (uncentered).
    random_state
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_components: int | None,
        *,
        zero_center: bool = True,
        random_state: int | None = 0,
    ) -> None:
        self.n_components = n_components
        self.zero_center = zero_center
        self.random_state = random_state

    def fit(self, x: spmatrix) -> Self:
        """
        Fit the PCA model.

        Parameters
        ----------
        x
            Sparse matrix of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        from ._helper import _check_matrix_for_zero_genes

        if self.n_components is None:
            n_rows = x.shape[0]
            n_cols = x.shape[1]
            self.n_components_ = min(n_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        _check_matrix_for_zero_genes(x)
        self.n_samples_ = x.shape[0]
        self.n_features_in_ = x.shape[1] if x.ndim == 2 else 1
        self.dtype = x.dtype

        # Compute mean if zero-centering
        if self.zero_center:
            self.mean_, _ = _get_mean_var(x, axis=0)
            self.mean_ = self.mean_.astype(x.dtype)
        else:
            self.mean_ = None

        # For Lanczos SVD with mean-centering, use implicit centering
        if self.zero_center:
            x_centered = MeanCenteredOperator(x, self.mean_)
            U, S, Vt = gpu_sparse_svds(
                x_centered,
                k=self.n_components_,
                random_state=self.random_state,
            )
        else:
            U, S, Vt = gpu_sparse_svds(
                x,
                k=self.n_components_,
                random_state=self.random_state,
            )

        # Components are the right singular vectors (Vt)
        self.components_ = Vt

        # Explained variance from singular values: var = s^2 / (n_samples - 1)
        self.explained_variance_ = (S**2) / (self.n_samples_ - 1)

        # Compute total variance for variance ratio
        if self.zero_center:
            _, var_x = _get_mean_var(x, axis=0)
            total_variance = cp.sum(var_x)
        else:
            if hasattr(x, "data"):
                total_variance = cp.sum(x.data**2) / (self.n_samples_ - 1)
            else:
                total_variance = cp.sum(x**2) / (self.n_samples_ - 1)

        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X: spmatrix) -> cp.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X
            Sparse matrix of shape (n_samples, n_features).

        Returns
        -------
        X_new : cp.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        return self._transform_cupy(X)

    def _transform_cupy(self, X: spmatrix) -> cp.ndarray:
        """CuPy implementation of transform."""
        if self.zero_center and self.mean_ is not None:
            # X_centered @ V.T = X @ V.T - mean @ V.T
            X_transformed = X.dot(self.components_.T)
            mean_projection = cp.dot(self.mean_, self.components_.T)
            X_transformed -= mean_projection
            return X_transformed
        else:
            return X.dot(self.components_.T)

    def fit_transform(self, X, y=None):
        """Fit the model and apply dimensionality reduction."""
        return self.fit(X).transform(X)
