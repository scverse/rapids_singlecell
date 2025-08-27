from __future__ import annotations

import cupy as cp


@cp.fuse
def _get_factor(o_k: cp.ndarray, ridge_lambda: float) -> cp.ndarray:
    return 1 / (o_k + ridge_lambda)


@cp.fuse
def _get_pen(e: cp.ndarray, o: cp.ndarray, theta: cp.ndarray) -> cp.ndarray:
    return cp.power(cp.divide(e + 1, o + 1), theta)


# @cp.fuse
# def _calc_R(term: float, dotproduct: cp.ndarray) -> cp.ndarray:
#    return cp.exp(term * (1 - dotproduct))


@cp.fuse
def _log_div_OE(o: cp.ndarray, e: cp.ndarray) -> cp.ndarray:
    return o * cp.log((o + 1) / (e + 1))


_entropy_kernel = cp.ReductionKernel(
    "T x",
    "T y",
    "x * logf(x + 1e-12)",
    "a + b",
    "y = a",
    "0",
    "entropy_reduce",
)


calc_R = cp.ElementwiseKernel(
    "T term, T x",
    "T y",
    "y = exp(term * (T(1) - x));",
    "calc_R",
    no_return=True,
)


def _calc_R(term: float, R: cp.ndarray) -> cp.ndarray:
    calc_R(term, R, R)
    return R
