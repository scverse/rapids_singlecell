from __future__ import annotations

import cupy as cp


@cp.fuse
def _get_factor(O_k: cp.ndarray, ridge_lambda: float) -> cp.ndarray:
    return 1 / (O_k + ridge_lambda)


@cp.fuse
def _get_pen(E: cp.ndarray, O: cp.ndarray, theta: cp.ndarray) -> cp.ndarray:
    return cp.power(cp.divide(E + 1, O + 1), theta)


@cp.fuse
def _calc_R(term: cp.ndarray, mm: cp.ndarray) -> cp.ndarray:
    return cp.exp(term * (1 - mm))


@cp.fuse
def _div_clip(X: cp.ndarray, norm: cp.ndarray) -> cp.ndarray:
    return X / cp.clip(norm, a_min=1e-12, a_max=cp.inf)


@cp.fuse
def _log_div_OE(O: cp.ndarray, E: cp.ndarray) -> cp.ndarray:
    return O * cp.log((O + 1) / (E + 1))


@cp.fuse
def _R_multi_m(R: cp.ndarray, other: cp.ndarray) -> cp.ndarray:
    return R * 2 * (1 - other)
