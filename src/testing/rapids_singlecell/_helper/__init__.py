"""Like fixtures, but more flexible"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import pytest
from anndata.tests.helpers import as_dense_dask_array, as_sparse_dask_array, asarray
from cupyx.scipy import sparse as cusparse
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    from _pytest.mark.structures import ParameterSet


def param_with(
    at: ParameterSet,
    *,
    marks: Iterable[pytest.Mark | pytest.MarkDecorator] = (),
    id: str | None = None,
) -> ParameterSet:
    return pytest.param(*at.values, marks=[*at.marks, *marks], id=id or at.id)


MAP_ARRAY_TYPES: dict[
    tuple[Literal["mem", "dask"], Literal["dense", "sparse"]],
    tuple[ParameterSet, ...],
] = {
    ("mem", "dense"): (pytest.param(asarray, id="numpy_ndarray"),),
    ("mem", "sparse"): (
        pytest.param(sparse.csr_matrix, id="scipy_csr"),
        pytest.param(sparse.csc_matrix, id="scipy_csc"),
    ),
}

ARRAY_TYPES_MEM = tuple(
    at for (strg, _), ats in MAP_ARRAY_TYPES.items() if strg == "mem" for at in ats
)


def as_sparse_cupy_dask_array(X):
    da = as_sparse_dask_array(X)
    da = da.rechunk((da.shape[0] // 2, da.shape[1]))
    da = da.map_blocks(cusparse.csr_matrix, dtype=X.dtype)
    return da


def as_dense_cupy_dask_array(X):
    X = as_dense_dask_array(X)
    X = X.map_blocks(cp.array)
    X = X.rechunk((X.shape[0] // 2, X.shape[1]))
    return X
