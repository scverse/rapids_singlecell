from __future__ import annotations

from scanpy.get import _check_mask, _get_obs_rep, _set_obs_rep

from ._aggregated import aggregate
from ._anndata import anndata_to_CPU, anndata_to_GPU
