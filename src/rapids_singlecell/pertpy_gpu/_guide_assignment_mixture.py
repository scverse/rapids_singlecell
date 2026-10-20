"""Poisson-Gaussian mixture model for gRNA assignment on the GPU.

Mirrors ``pertpy.preprocessing._guide_rna_mixture``, which implements the model of ``crispat.ga_poisson_gauss`` (Velten group, https://github.com/velten-group/crispat).
For every guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)`` and ``lam ~ LogNormal(0, 1)``.
The MAP estimate is found by multi-start EM on the distinct count values of every guide, for all guides and starts in parallel with CuPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupyx.scipy.special import gammaln
from scipy.special import gammaln as _np_gammaln

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))
_SCALE_FLOOR = 1e-3
_WEIGHT_FLOOR = 1e-6
_INIT_QUANTILES = (0.5, 0.75, 0.9, 0.97, 0.99)
_PRIOR_INIT_SAMPLES = 15
_NEWTON_STEPS = 12
_CHUNK_ELEMENTS = 2**22
_CONVERGENCE_CHECK_EVERY = 25
_LOG_SCALE_FLOOR = float(np.log(_SCALE_FLOOR))


@dataclass
class PoissonGaussFit:
    """MAP estimates per guide, ``nan`` where no admissible solution exists."""

    poisson_rate: np.ndarray
    gaussian_mean: np.ndarray
    gaussian_std: np.ndarray
    mix_probs: np.ndarray
    final_loss: np.ndarray


@cp.fuse()
def _mixture_terms(values, log_gamma, rate, mu, scale, w_norm):  # noqa: PLR0917
    """Per-value log mixture density and Gaussian responsibility."""
    log_pois = values * cp.log(rate) - rate - log_gamma
    log_norm = -_LOG_SQRT_2PI - cp.log(scale) - 0.5 * cp.square((values - mu) / scale)
    log_norm_weighted = cp.log(w_norm) + log_norm
    log_mix = cp.logaddexp(log_norm_weighted, cp.log1p(-w_norm) + log_pois)
    return log_mix, cp.exp(log_norm_weighted - log_mix)


@cp.fuse()
def _log_prior(rate, mu, scale, w_norm):
    return (
        -0.1 * cp.log1p(-w_norm)
        - 0.9 * cp.log(w_norm)
        - 0.5 * cp.square((mu - 3.0) / 2.0)
        - cp.log(scale)
        - 0.5 * cp.square(cp.log(scale) - 2.0)
        - cp.log(rate)
        - 0.5 * cp.square(cp.log(rate))
    )


def _neg_log_joint(
    rate: cp.ndarray,
    mu: cp.ndarray,
    scale: cp.ndarray,
    w_norm: cp.ndarray,
    *,
    values: cp.ndarray,
    weights: cp.ndarray,
    log_gamma: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Negative log joint per start and guide, with the Gaussian responsibilities.

    Parameters have shape ``[starts, guides]``, the data ``[guides, values]``.
    Additive prior constants are dropped.
    """
    log_mix, resp = _mixture_terms(
        values,
        log_gamma,
        rate[..., None],
        mu[..., None],
        scale[..., None],
        w_norm[..., None],
    )
    loss = -((weights * log_mix).sum(axis=-1) + _log_prior(rate, mu, scale, w_norm))
    return loss, resp


@cp.fuse()
def _solve_log_scale(sum_squares, n_norm):
    """MAP log scale of the Gaussian: the root of a convex decreasing function."""
    v = 0.5 * cp.log(cp.maximum(sum_squares, 1e-12) / cp.maximum(n_norm, 1e-12))
    v = cp.minimum(cp.maximum(v, _LOG_SCALE_FLOOR), 10.0)
    for _ in range(_NEWTON_STEPS):
        e = sum_squares * cp.exp(-2.0 * v)
        f = e - (v - 2.0) - n_norm - 1.0
        v = cp.minimum(cp.maximum(v - f / (-2.0 * e - 1.0), _LOG_SCALE_FLOOR), 10.0)
    return v


@cp.fuse()
def _solve_log_rate(sum_values, n_pois):
    """MAP log rate of the Poisson: the root of a concave decreasing function."""
    log_rate = cp.log(cp.maximum(sum_values, 1e-12) / cp.maximum(n_pois, 1e-12))
    log_rate = cp.minimum(cp.maximum(log_rate, -20.0), 10.0)
    for _ in range(_NEWTON_STEPS):
        e = n_pois * cp.exp(log_rate)
        g = sum_values - 1.0 - log_rate - e
        log_rate = cp.minimum(cp.maximum(log_rate - g / (-1.0 - e), -20.0), 10.0)
    return log_rate


def _em_step(
    params: tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray],
    values: cp.ndarray,
    weights: cp.ndarray,
    log_gamma: cp.ndarray,
) -> tuple[tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray], cp.ndarray]:
    """One MAP-EM step; returns the updated parameters and the loss of the old ones."""
    rate, mu, scale, w_norm = params
    loss, resp = _neg_log_joint(
        rate, mu, scale, w_norm, values=values, weights=weights, log_gamma=log_gamma
    )
    w_resp = weights * resp
    n_norm = w_resp.sum(axis=-1)
    n_pois = weights.sum(axis=-1) - n_norm
    w_new = cp.clip(
        (n_norm - 0.9) / (n_norm + n_pois - 1.0), _WEIGHT_FLOOR, 1.0 - _WEIGHT_FLOOR
    )
    precision = 1.0 / cp.square(scale)
    sum_resp_values = (w_resp * values).sum(axis=-1)
    mu_new = (sum_resp_values * precision + 0.75) / (n_norm * precision + 0.25)
    sum_squares = (w_resp * cp.square(values - mu_new[..., None])).sum(axis=-1)
    scale_new = cp.maximum(cp.exp(_solve_log_scale(sum_squares, n_norm)), _SCALE_FLOOR)
    sum_rest_values = (weights * values).sum(axis=-1) - sum_resp_values
    rate_new = cp.exp(_solve_log_rate(sum_rest_values, n_pois))
    return (rate_new, mu_new, scale_new, w_new), loss


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
    rng: cp.random.Generator,
    *,
    n_init_seeds: int,
    n_iter: int,
    tol: float,
) -> tuple[cp.ndarray, ...]:
    """Fit all starts of a chunk of guides and keep the best admissible solution per guide.

    Label-switched optima (Gaussian below the Poisson) and solutions pinned at the weight or scale floor are inadmissible.
    """
    log_gamma = gammaln(values + 1.0).astype(cp.float32, copy=False)
    inits = cp.concatenate(
        [
            _prior_median_inits(rng, values.shape[0], n_init_seeds),
            _data_inits(values, weights),
        ]
    )
    params = tuple(cp.ascontiguousarray(inits[..., k]) for k in range(4))
    previous = cp.full(params[0].shape, cp.inf, dtype=cp.float32)
    current = cp.full(params[0].shape, 1e30, dtype=cp.float32)
    active = cp.ones(params[0].shape, dtype=cp.bool_)
    for iteration in range(1, n_iter + 1):
        new_params, loss = _em_step(params, values, weights, log_gamma)
        params = tuple(
            cp.where(active, new, old) for new, old in zip(new_params, params)
        )
        previous = cp.where(active, current, previous)
        current = cp.where(active, loss, current)
        active &= cp.abs(previous - current) > tol * cp.abs(current)
        if iteration % _CONVERGENCE_CHECK_EVERY == 0 and not bool(active.any()):
            break
    rate, mu, scale, w_norm = params
    loss, _ = _neg_log_joint(
        rate, mu, scale, w_norm, values=values, weights=weights, log_gamma=log_gamma
    )
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

    Guides are grouped by their number of distinct values and fit in chunks of bounded size, so peak memory does not grow with the number of guides.

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
    n_values = cp.asnumpy((weights > 0).sum(axis=1))
    bucket = np.ceil(np.log2(np.maximum(n_values, 2))).astype(int)

    out = {
        name: np.full(n_guides, np.nan)
        for name in ("rate", "mu", "scale", "w_norm", "loss")
    }
    for chunk_id, b in enumerate(np.unique(bucket)):
        guides = np.flatnonzero(bucket == b)
        width = min(2**b, values.shape[1])
        rows = 2 ** int(
            np.ceil(
                np.log2(min(guides.size, max(_CHUNK_ELEMENTS // (n_starts * width), 1)))
            )
        )
        for part, chunk in enumerate(
            np.array_split(guides, int(np.ceil(guides.size / rows)))
        ):
            chunk_values = cp.zeros((rows, width), dtype=cp.float32)
            chunk_weights = cp.zeros((rows, width), dtype=cp.float32)
            chunk_idx = cp.asarray(chunk)
            chunk_values[: chunk.size] = values[chunk_idx, :width]
            chunk_weights[: chunk.size] = weights[chunk_idx, :width]
            rng = cp.random.default_rng(seed * 1_000_003 + chunk_id * 2**16 + part)
            rate, mu, scale, w_norm, loss, ok = (
                cp.asnumpy(a)[: chunk.size]
                for a in _fit_chunk(
                    chunk_values,
                    chunk_weights,
                    rng,
                    n_init_seeds=n_init_seeds,
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
    thresholds = np.full(len(fit.poisson_rate), np.nan, dtype=np.float64)
    for g, max_c in enumerate(max_counts):
        if max_c < 1 or np.isnan(fit.gaussian_mean[g]):
            continue
        counts = np.arange(1, int(max_c) + 1, dtype=np.float64)
        log_counts = np.log2(counts)
        mu = float(fit.gaussian_mean[g])
        scale = float(fit.gaussian_std[g])
        rate = float(fit.poisson_rate[g])
        log_norm = (
            -_LOG_SQRT_2PI - np.log(scale) - 0.5 * ((log_counts - mu) / scale) ** 2
        )
        log_pois = log_counts * np.log(rate) - rate - _np_gammaln(log_counts + 1.0)
        log_num = np.log(fit.mix_probs[g, 1]) + log_norm
        log_den = np.logaddexp(log_num, np.log(fit.mix_probs[g, 0]) + log_pois)
        above = np.flatnonzero(np.exp(log_num - log_den) > 0.5)
        if above.size > 0:
            thresholds[g] = float(counts[above[0]])
    return thresholds
