from __future__ import annotations

import cupy as cp


@cp.fuse
def _get_factor(o_k: cp.ndarray, ridge_lambda: float) -> cp.ndarray:
    return 1 / (o_k + ridge_lambda)


@cp.fuse
def _calc_R(term: float, dotproduct: cp.ndarray) -> cp.ndarray:
    return cp.exp(term * (1 - dotproduct))
