from __future__ import annotations

import cuml.internals.logger as logger

from . import dcg, get, gr, pp, tl
from ._version import __version__

logger.set_level(2)
