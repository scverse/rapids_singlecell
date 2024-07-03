from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupyx.scipy import sparse
from scanpy._utils import get_random_state

from rapids_singlecell.preprocessing._utils import _get_mean_var

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rapids_singlecell._utils import AnyRandom


def sparse_multiply(
    E: sparse.csr_matrix | sparse.csc_matrix | NDArray[np.float64],
    a: float | int | NDArray[np.float64],
) -> sparse.csr_matrix | sparse.csc_matrix:
    """multiply each row of E by a scalar"""

    nrow = E.shape[0]
    w = sparse.dia_matrix((a, 0), shape=(nrow, nrow), dtype=a.dtype)
    # w.setdiag(a)
    r = w @ E
    if isinstance(r, (cp.ndarray)):
        return sparse.csc_matrix(r)
    return r


def sparse_zscore(
    E: sparse.csr_matrix | sparse.csc_matrix,
    *,
    gene_mean: NDArray[np.float64] | None = None,
    gene_stdev: NDArray[np.float64] | None = None,
) -> sparse.csr_matrix | sparse.csc_matrix:
    """z-score normalize each column of E"""
    if gene_mean is None or gene_stdev is None:
        gene_means, gene_stdevs = _get_mean_var(E, axis=0)
        gene_stdevs = cp.sqrt(gene_stdevs)
    return sparse_multiply(cp.asarray((E - gene_mean).T), 1 / gene_stdev).T


def subsample_counts(
    E: sparse.csr_matrix | sparse.csc_matrix,
    *,
    rate: float,
    original_totals,
    random_seed: AnyRandom = 0,
) -> tuple[sparse.csr_matrix | sparse.csc_matrix, NDArray[np.int64]]:
    if rate < 1:
        random_seed = get_random_state(random_seed)
        dtype = E.dtype
        E.data = cp.array(
            random_seed.binomial(np.round(E.data.get()).astype(int), rate), dtype=dtype
        )
        current_totals = E.sum(1).ravel()
        unsampled_orig_totals = original_totals - current_totals
        unsampled_downsamp_totals = cp.random.binomial(
            cp.round(unsampled_orig_totals).astype(int),
            rate,
            dtype=dtype,
        )
        final_downsamp_totals = current_totals + unsampled_downsamp_totals
    else:
        final_downsamp_totals = original_totals
    return E, final_downsamp_totals
