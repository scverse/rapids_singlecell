"""
Linear operators for implicit mean-centering of sparse matrices.

These operators avoid materializing the dense centered matrix by computing
the centering on-the-fly during matrix-vector products.
"""

from __future__ import annotations

import cupy as cp


class MeanCenteredOperator:
    """
    Linear operator for mean-centered sparse matrix.

    For matrix X with column means, computes products with (X - 1*mean.T)
    without explicitly forming the dense centered matrix.

    Parameters
    ----------
    X : sparse matrix
        The sparse matrix to center.
    mean : cupy.ndarray
        Column means of shape (n_features,).
    """

    def __init__(self, X, mean: cp.ndarray) -> None:
        self.X = X
        self.mean = mean
        self.shape = X.shape
        self.dtype = X.dtype

    def dot(self, v: cp.ndarray) -> cp.ndarray:
        """
        Compute (X - 1*mean.T) @ v.

        For vector v: X @ v - (mean @ v) (broadcasted)
        For matrix v: X @ v - outer(ones, mean @ v)
        """
        Xv = self.X.dot(v)
        mean_v = cp.dot(self.mean, v)
        if v.ndim == 1:
            return Xv - mean_v
        else:
            return Xv - mean_v[cp.newaxis, :]

    def __matmul__(self, other):
        return self.dot(other)

    @property
    def T(self):
        """Return transposed operator."""
        return MeanCenteredOperatorT(self.X, self.mean)


class MeanCenteredOperatorT:
    """Transposed mean-centered operator."""

    def __init__(self, X, mean: cp.ndarray) -> None:
        self.X = X
        self.mean = mean
        self.shape = (X.shape[1], X.shape[0])
        self.dtype = X.dtype

    def dot(self, v: cp.ndarray) -> cp.ndarray:
        """
        Compute (X - 1*mean.T).T @ v = X.T @ v - mean * sum(v).
        """
        XTv = self.X.T.dot(v)
        if v.ndim == 1:
            sum_v = cp.sum(v)
            return XTv - self.mean * sum_v
        else:
            sum_v = cp.sum(v, axis=0)
            return XTv - cp.outer(self.mean, sum_v)

    def __matmul__(self, other):
        return self.dot(other)
