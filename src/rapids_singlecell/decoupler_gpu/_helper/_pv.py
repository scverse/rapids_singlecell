from __future__ import annotations

import cupy as cp


def fdr_bh_axis1(ps):
    """
    Benjamini–Hochberg adjusted p-values along axis=1 (rows).
    ps: cupy.ndarray (n_rows, n_tests), values in [0, 1].
    Returns: cupy.ndarray of same shape.
    """
    ps = cp.asarray(ps, dtype=cp.float64)
    if ps.ndim != 2:
        raise ValueError("ps must be 2D (n_rows, n_tests) for axis=1.")
    if not cp.issubdtype(ps.dtype, cp.number):
        raise ValueError("`ps` must be numeric.")
    if not bool((ps == cp.clip(ps, 0, 1)).all()):
        raise ValueError("`ps` must be within [0, 1].")

    n_rows, m = ps.shape
    if m <= 1:
        return ps.copy()

    # sort each row
    order = cp.argsort(ps, axis=1)
    ps_sorted = cp.take_along_axis(ps, order, axis=1)

    # BH scale: p_(i) * m / i
    i = cp.arange(1, m + 1, dtype=ps.dtype)
    ps_bh = ps_sorted * (m / i)

    # reverse cumulative min across columns (no ufunc.accumulate in CuPy)
    ps_rev = ps_bh[:, ::-1].copy()
    for j in range(1, m):
        ps_rev[:, j] = cp.minimum(ps_rev[:, j], ps_rev[:, j - 1])
    ps_monotone = ps_rev[:, ::-1]

    # unsort back to original column order
    inv_order = cp.argsort(order, axis=1)
    ps_adj = cp.take_along_axis(ps_monotone, inv_order, axis=1)

    return cp.clip(ps_adj, 0, 1).get().astype(cp.float32)
