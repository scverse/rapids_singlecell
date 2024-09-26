"""Like fixtures, but more flexible"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from anndata.tests.helpers import asarray
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
