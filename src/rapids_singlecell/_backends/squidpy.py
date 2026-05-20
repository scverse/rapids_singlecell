from __future__ import annotations

from rapids_singlecell.squidpy_gpu import co_occurrence, ligrec, spatial_autocorr

name = "rapids_singlecell"
aliases = ["rapids-singlecell", "rsc", "cuda"]

__all__ = ["co_occurrence", "ligrec", "spatial_autocorr"]
