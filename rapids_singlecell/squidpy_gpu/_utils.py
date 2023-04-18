import numpy as np
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Union,
)
from scipy import stats
from scipy.sparse import spmatrix

### Taken from squidpy: https://github.com/scverse/squidpy/blob/main/squidpy/gr/_ppatterns.py
def _p_value_calc(
    score: np.ndarray,
    sims: Union[np.ndarray, None],
    weights: Union[spmatrix, np.ndarray],
    params: Dict[str, Any],
):
    """
    Handle p-value calculation for spatial autocorrelation function.
    Parameters
    ----------
    score
        (n_features,).
    sims
        (n_simulations, n_features).
    params
        Object to store relevant function parameters.
    Returns
    -------
    pval_norm
        p-value under normality assumption
    pval_sim
        p-values based on permutations
    pval_z_sim
        p-values based on standard normal approximation from permutations
    """
    p_norm, var_norm = _analytic_pval(score, weights, params)
    results = {"pval_norm": p_norm, "var_norm": var_norm}

    if sims is None:
        return results

    n_perms = sims.shape[0]
    large_perm = (sims >= score).sum(axis=0)
    # subtract total perm for negative values
    large_perm[(n_perms - large_perm) < large_perm] = n_perms - large_perm[(n_perms - large_perm) < large_perm]
    # get p-value based on permutation
    p_sim: np.ndarray = (large_perm + 1) / (n_perms + 1)

    # get p-value based on standard normal approximation from permutations
    e_score_sim = sims.sum(axis=0) / n_perms
    se_score_sim = sims.std(axis=0)
    z_sim = (score - e_score_sim) / se_score_sim
    p_z_sim = np.empty(z_sim.shape)

    p_z_sim[z_sim > 0] = 1 - stats.norm.cdf(z_sim[z_sim > 0])
    p_z_sim[z_sim <= 0] = stats.norm.cdf(z_sim[z_sim <= 0])

    var_sim = np.var(sims, axis=0)

    results["pval_z_sim"] = p_z_sim
    results["pval_sim"] = p_sim
    results["var_sim"] = var_sim

    return results


def _analytic_pval(score: np.ndarray, g: Union[spmatrix, np.ndarray], params: Dict[str, Any]):
    """
    Analytic p-value computation.
    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    n2 = n * n
    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02

    Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
    seScore_norm = Vscore_norm ** (1 / 2.0)

    z_norm = (score - params["expected"]) / seScore_norm
    p_norm = np.empty(score.shape)
    p_norm[z_norm > 0] = 1 - stats.norm.cdf(z_norm[z_norm > 0])
    p_norm[z_norm <= 0] = stats.norm.cdf(z_norm[z_norm <= 0])

    if params["two_tailed"]:
        p_norm *= 2.0

    return p_norm, Vscore_norm


def _g_moments(w: Union[spmatrix, np.ndarray]):
    """
    Compute moments of adjacency matrix for analytic p-value calculation.
    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    # s0
    s0 = w.sum()

    # s1
    t = w.transpose() + w
    t2 = t.multiply(t)  # type: ignore[union-attr]
    s1 = t2.sum() / 2.0

    # s2
    s2array: np.ndarray = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

    return s0, s1, s2
