"""Poisson-Gaussian mixture model for gRNA assignment on the GPU.

Mirrors ``pertpy.preprocessing._guide_rna_mixture``, which implements the model of ``crispat.ga_poisson_gauss`` (Velten group, https://github.com/velten-group/crispat).
For every guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)`` and ``lam ~ LogNormal(0, 1)``.
The MAP estimate is found by multi-start EM on the distinct count values of every guide, with one CUDA block per guide and start.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _guide_assignment_cuda as _ga

_N_QUANTILE_STARTS = 5
_PRIOR_INIT_SAMPLES = 15
_N_ESTIMATES = 6


@dataclass
class PoissonGaussFit:
    """MAP estimates and assignment thresholds per guide, ``nan`` where no admissible solution exists."""

    poisson_rate: np.ndarray
    gaussian_mean: np.ndarray
    gaussian_std: np.ndarray
    mix_probs: np.ndarray
    final_loss: np.ndarray
    threshold: np.ndarray


def fit_poisson_gauss_mixture(
    values: cp.ndarray,
    weights: cp.ndarray,
    *,
    n_values: cp.ndarray,
    max_counts: cp.ndarray,
    n_iter: int = 1000,
    n_init_seeds: int = 10,
    seed: int = 2024,
    tol: float = 1e-7,
) -> PoissonGaussFit:
    """Fit Poisson-Gaussian mixtures for many guides in parallel by multi-start MAP-EM.

    Parameters
    ----------
    values
        Sorted log2-transformed distinct non-zero counts per guide of shape ``[n_guides, width]``, zero padded.
    weights
        Number of cells with each value, zero for padding.
    n_values
        Number of distinct values per guide.
    max_counts
        Largest count per guide, the upper end of the threshold scan.
    n_iter
        Maximum EM iterations per start.
    n_init_seeds
        Number of prior-median initialisations per guide, used in addition to five quantile-split initialisations.
    seed
        Random seed for the prior-median initialisations.
    tol
        Relative change of the loss below which EM stops.

    Returns
    -------
    MAP estimates and thresholds for all guides.
    """
    values = cp.ascontiguousarray(values, dtype=cp.float32)
    weights = cp.ascontiguousarray(weights, dtype=cp.float32)
    n_values = cp.asarray(n_values, dtype=cp.int32)
    max_counts = cp.asarray(max_counts, dtype=cp.int32)
    n_guides = values.shape[0]
    n_starts = n_init_seeds + _N_QUANTILE_STARTS
    out = cp.full((n_guides, _N_ESTIMATES), cp.nan, dtype=cp.float32)
    if n_guides:
        rng = cp.random.default_rng(seed)
        draws = [
            rng.standard_normal(
                (n_init_seeds, _PRIOR_INIT_SAMPLES, n_guides), dtype=cp.float32
            )
            for _ in range(3)
        ]
        params = cp.empty((n_starts, n_guides, 4), dtype=cp.float32)
        loss = cp.empty((n_starts, n_guides), dtype=cp.float32)
        stream = cp.cuda.get_current_stream().ptr
        _ga.fit_poisson_gauss(
            values,
            weights,
            n_values,
            *draws,
            params,
            loss,
            n_iter=n_iter,
            tol=tol,
            stream=stream,
        )
        _ga.select_threshold(params, loss, max_counts, out, stream=stream)
    estimates = cp.asnumpy(out).astype(np.float64)
    return PoissonGaussFit(
        poisson_rate=estimates[:, 0],
        gaussian_mean=estimates[:, 1],
        gaussian_std=estimates[:, 2],
        mix_probs=np.stack([1.0 - estimates[:, 3], estimates[:, 3]], axis=-1),
        final_loss=estimates[:, 4],
        threshold=estimates[:, 5],
    )
