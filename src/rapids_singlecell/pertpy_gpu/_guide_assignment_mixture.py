"""Poisson-Gaussian mixture model for gRNA assignment on the GPU.

Mirrors ``pertpy.preprocessing._guide_rna_mixture``, which implements the model of ``crispat.ga_poisson_gauss`` (Velten group, https://github.com/velten-group/crispat).
For every guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)`` and ``lam ~ LogNormal(0, 1)``.
The MAP estimate is found by multi-start EM on the distinct count values of every guide, with one CUDA block per guide and start.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupyx.scipy.special import gammaln

from rapids_singlecell._cuda import _guide_assignment_cuda as _ga

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))
_SCALE_FLOOR = 1e-3
_WEIGHT_FLOOR = 1e-6
_INIT_QUANTILES = (0.5, 0.75, 0.9, 0.97, 0.99)
_PRIOR_INIT_SAMPLES = 15
_CHUNK_ELEMENTS = 2**22
_SCAN_ELEMENTS = 2**22


@dataclass
class PoissonGaussFit:
    """MAP estimates per guide, ``nan`` where no admissible solution exists."""

    poisson_rate: np.ndarray
    gaussian_mean: np.ndarray
    gaussian_std: np.ndarray
    mix_probs: np.ndarray
    final_loss: np.ndarray


def _prior_median_inits(
    rng: cp.random.Generator, n_guides: int, n_starts: int
) -> cp.ndarray:
    """Initialisations as in crispat: prior medians and equal weights."""
    shape = (n_starts, _PRIOR_INIT_SAMPLES, n_guides)
    rate = cp.exp(cp.median(rng.standard_normal(shape, dtype=cp.float32), axis=1))
    mu = cp.median(3.0 + 2.0 * rng.standard_normal(shape, dtype=cp.float32), axis=1)
    scale = cp.exp(
        cp.median(2.0 + rng.standard_normal(shape, dtype=cp.float32), axis=1)
    )
    w_norm = cp.full((n_starts, n_guides), 0.5, dtype=cp.float32)
    return cp.stack([rate, mu, scale, w_norm], axis=-1)


def _data_inits(values: cp.ndarray, weights: cp.ndarray) -> cp.ndarray:
    """Initialisations that split every guide at a quantile, Poisson below, Gaussian above."""
    total = cp.maximum(weights.sum(axis=1), 1.0)
    cumulative = cp.cumsum(weights, axis=1) / total[:, None]
    inits = []
    for q in _INIT_QUANTILES:
        upper = (cumulative > q) & (weights > 0)
        w_up = cp.where(upper, weights, 0.0)
        w_low = weights - w_up
        n_up = w_up.sum(axis=1)
        n_low = w_low.sum(axis=1)
        rate = cp.maximum((w_low * values).sum(axis=1) / cp.maximum(n_low, 1.0), 0.05)
        mu = cp.where(
            n_up > 0,
            (w_up * values).sum(axis=1) / cp.maximum(n_up, 1.0),
            values.max(axis=1),
        )
        variance = (w_up * cp.square(values - mu[:, None])).sum(axis=1) / cp.maximum(
            n_up, 1.0
        )
        scale = cp.maximum(cp.sqrt(variance), 0.5)
        w_norm = cp.clip(n_up / total, 0.01, 0.9)
        inits.append(cp.stack([rate, mu, scale, w_norm], axis=-1))
    return cp.stack(inits).astype(cp.float32)


def _fit_chunk(
    values: cp.ndarray,
    weights: cp.ndarray,
    log_gamma: cp.ndarray,
    inits: cp.ndarray,
    *,
    n_iter: int,
    tol: float,
) -> tuple[cp.ndarray, ...]:
    """Fit all starts of a group of guides and keep the best admissible solution per guide.

    Label-switched optima (Gaussian below the Poisson) and solutions pinned at the weight or scale floor are inadmissible.
    """
    params = cp.full_like(inits, cp.nan)
    loss = cp.full(inits.shape[:2], cp.inf, dtype=cp.float32)
    _ga.fit_poisson_gauss(
        values,
        weights,
        log_gamma,
        inits,
        params,
        loss,
        n_guides=values.shape[0],
        n_iter=n_iter,
        tol=tol,
        stream=cp.cuda.get_current_stream().ptr,
    )
    rate, mu, scale, w_norm = (cp.ascontiguousarray(params[..., k]) for k in range(4))
    admissible = (
        cp.isfinite(loss)
        & (mu > rate)
        & (w_norm > 2 * _WEIGHT_FLOOR)
        & (scale > 2 * _SCALE_FLOOR)
    )
    best = cp.argmin(cp.where(admissible, loss, cp.inf), axis=0)[None]
    return tuple(
        cp.take_along_axis(a, best, axis=0)[0]
        for a in (rate, mu, scale, w_norm, loss, admissible)
    )


def fit_poisson_gauss_mixture(
    values: cp.ndarray,
    weights: cp.ndarray,
    *,
    n_iter: int = 1000,
    n_init_seeds: int = 10,
    seed: int = 2024,
    tol: float = 1e-7,
) -> PoissonGaussFit:
    """Fit Poisson-Gaussian mixtures for many guides in parallel by multi-start MAP-EM.

    Guides are grouped by their number of distinct values and fit in chunks, which bounds the zero padding of each launch.

    Parameters
    ----------
    values
        Sorted log2-transformed distinct non-zero counts per guide of shape ``[n_guides, n_values]``, zero padded.
    weights
        Number of cells with each value, zero for padding.
    n_iter
        Maximum EM iterations per start.
    n_init_seeds
        Number of prior-median initialisations per guide, used in addition to quantile-split initialisations.
    seed
        Random seed for the prior-median initialisations.
    tol
        Relative change of the loss below which EM stops.

    Returns
    -------
    MAP estimates for all guides.
    """
    values = cp.asarray(values, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)
    n_guides = values.shape[0]
    n_starts = n_init_seeds + len(_INIT_QUANTILES)
    log_gamma = gammaln(values + 1.0).astype(cp.float32, copy=False)
    data_inits = _data_inits(values, weights)
    n_values = cp.asnumpy((weights > 0).sum(axis=1))
    bucket = np.ceil(np.log2(np.maximum(n_values, 2))).astype(int)

    out = {
        name: np.full(n_guides, np.nan)
        for name in ("rate", "mu", "scale", "w_norm", "loss")
    }
    for chunk_id, b in enumerate(np.unique(bucket)):
        guides = np.flatnonzero(bucket == b)
        width = min(2**b, values.shape[1])
        # rows and parts also define the random starts; changing them changes the fit
        rows = 2 ** int(
            np.ceil(
                np.log2(min(guides.size, max(_CHUNK_ELEMENTS // (n_starts * width), 1)))
            )
        )
        for part, chunk in enumerate(
            np.array_split(guides, int(np.ceil(guides.size / rows)))
        ):
            chunk_gpu = cp.asarray(chunk)
            rng = cp.random.default_rng(seed * 1_000_003 + chunk_id * 2**16 + part)
            inits = cp.concatenate(
                [
                    _prior_median_inits(rng, rows, n_init_seeds)[:, : chunk.size],
                    data_inits[:, chunk_gpu],
                ]
            )
            rate, mu, scale, w_norm, loss, ok = (
                cp.asnumpy(a)
                for a in _fit_chunk(
                    values[chunk_gpu, :width],
                    weights[chunk_gpu, :width],
                    log_gamma[chunk_gpu, :width],
                    cp.ascontiguousarray(inits),
                    n_iter=n_iter,
                    tol=tol,
                )
            )
            fitted = chunk[ok]
            for name, estimate in zip(
                out, (rate, mu, scale, w_norm, loss), strict=True
            ):
                out[name][fitted] = estimate[ok]

    return PoissonGaussFit(
        poisson_rate=out["rate"],
        gaussian_mean=out["mu"],
        gaussian_std=out["scale"],
        mix_probs=np.stack([1.0 - out["w_norm"], out["w_norm"]], axis=-1),
        final_loss=out["loss"],
    )


def compute_count_thresholds(
    fit: PoissonGaussFit, max_counts: np.ndarray
) -> np.ndarray:
    """Per-guide assignment threshold matching crispat.

    For each guide, scans integer UMI counts ``t = 1, 2, ..., max_count`` and returns the smallest ``t`` for which ``P(Normal | log2(t)) > 0.5``.
    Returns ``np.nan`` if no such ``t`` exists within ``max_count`` or the guide has no fit.
    """
    max_counts = np.asarray(max_counts, dtype=np.int64)
    thresholds = np.full(len(max_counts), np.nan, dtype=np.float64)
    fitted = np.flatnonzero(np.isfinite(fit.gaussian_mean) & (max_counts >= 1))
    lengths = max_counts[fitted]
    chunk_of_guide = (np.cumsum(lengths) - lengths) // _SCAN_ELEMENTS
    for chunk in np.unique(chunk_of_guide):
        guides = fitted[chunk_of_guide == chunk]
        thresholds[guides] = _scan_thresholds(fit, guides, max_counts[guides])
    return thresholds


def _scan_thresholds(
    fit: PoissonGaussFit, guides: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Smallest count per guide with a Gaussian posterior above 0.5, scanned on the GPU."""
    lengths_gpu = cp.asarray(lengths)
    offsets = cp.cumsum(lengths_gpu) - lengths_gpu
    segment = cp.repeat(cp.arange(len(guides)), lengths_gpu)
    total = int(lengths.sum())
    counts = (cp.arange(total) - offsets[segment] + 1).astype(cp.float64)
    log_counts = cp.log2(counts)
    mu, scale, rate, w_pois, w_norm = (
        cp.asarray(a[guides])[segment]
        for a in (
            fit.gaussian_mean,
            fit.gaussian_std,
            fit.poisson_rate,
            fit.mix_probs[:, 0],
            fit.mix_probs[:, 1],
        )
    )
    log_norm = (
        -_LOG_SQRT_2PI - cp.log(scale) - 0.5 * cp.square((log_counts - mu) / scale)
    )
    log_pois = log_counts * cp.log(rate) - rate - gammaln(log_counts + 1.0)
    log_num = cp.log(w_norm) + log_norm
    log_den = cp.logaddexp(log_num, cp.log(w_pois) + log_pois)
    above = cp.flatnonzero(cp.exp(log_num - log_den) > 0.5)
    first = cp.full(len(guides), total, dtype=cp.int64)
    cp.minimum.at(first, segment[above], above)
    found = first < total
    result = np.full(len(guides), np.nan)
    result[cp.asnumpy(found)] = cp.asnumpy((first - offsets + 1)[found])
    return result
