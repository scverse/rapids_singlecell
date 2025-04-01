from __future__ import annotations

from typing import TYPE_CHECKING, Union

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from dask.array import Array as DaskArray

AnyRandom = Union[int, np.random.RandomState, None]  # noqa: UP007


ArrayTypes = Union[cp.ndarray, csc_matrix, csr_matrix]  # noqa: UP007
ArrayTypesDask = Union[cp.ndarray, csc_matrix, csr_matrix, DaskArray]  # noqa: UP007


def _get_logger_level(logger):
    for i in range(15):
        out = logger.should_log_for(i)
        if out:
            return i
