"""
This module will benchmark io of Scanpy readwrite operations

Things to test:

* Read time, write time
* Peak memory during io
* File sizes

Parameterized by:

* What method is being used
* What data is being included
* Size of data being used

Also interesting:

* io for views
* io for backed objects
* Reading dense as sparse, writing sparse as dense
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata
import scanpy as sc

from rapids_singlecell.get import anndata_to_GPU

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass
class Dataset:
    path: Path
    get: Callable[[], anndata.AnnData]


pbmc3k = Dataset(
    path=sc.settings.datasetdir / "pbmc3k_raw.h5ad", get=sc.datasets.pbmc3k
)


class ToGPUSuite:
    _data_dict = dict(pbmc3k=pbmc3k)
    params = _data_dict.keys()
    param_names = ["input_data"]

    def setup(self, input_data: str):
        self.path = self._data_dict[input_data].path
        self.data = self._data_dict[input_data].get()

    def time_to_gpu(self, *_):
        anndata_to_GPU(self.data)

    def peakmem_to_gpu(self, *_):
        anndata_to_GPU(self.data)

    def mem_to_gpu(self, *_):
        anndata_to_GPU(self.data)
