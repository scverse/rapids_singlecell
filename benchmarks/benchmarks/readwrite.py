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

import scanpy as sc

from rapids_singlecell.get import anndata_to_GPU

from .utils import track_peakmem


class ToGPUSuite:
    _data_dict = dict(obmc68k_reduced=sc.datasets.pbmc68k_reduced())
    params = _data_dict.keys()
    param_names = ["input_data"]

    def setup(self, input_data: str):
        self.adata = self._data_dict[input_data]

    def time_to_gpu(self, *_):
        anndata_to_GPU(self.adata)

    @track_peakmem
    def track_peakmem_to_gpu(self, *_):
        anndata_to_GPU(self.adata)
