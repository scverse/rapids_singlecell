"""Squidpy backend adapter for rapids_singlecell.

The dispatch decorator introspects the real RSC function signatures
(lazily imported on first access), so no need to duplicate them here.
"""

from __future__ import annotations

import importlib


class RscSquidpyBackend:
    """Backend adapter exposing rapids_singlecell GPU implementations to squidpy."""

    name = "rapids_singlecell"
    aliases = ["rapids-singlecell", "rsc", "cuda", "gpu"]

    # squidpy function name -> module that implements it
    _functions = {
        "spatial_autocorr": "rapids_singlecell.squidpy_gpu",
        "co_occurrence": "rapids_singlecell.squidpy_gpu",
        "ligrec": "rapids_singlecell.squidpy_gpu",
    }

    def __getattr__(self, name: str):
        if name in self._functions:
            func = getattr(importlib.import_module(self._functions[name]), name)
            setattr(self, name, func)  # cache on instance
            return func
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")
