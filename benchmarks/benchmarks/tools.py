"""
This module will benchmark tool operations in Scanpy
API documentation: https://scanpy.readthedocs.io/en/stable/api/tools.html
"""

from __future__ import annotations

import scanpy as sc

import rapids_singlecell as rsc

from .utils import track_peakmem


class ToolsSuite:
    _data_dict = dict(
        pbmc68k_reduced=sc.datasets.pbmc68k_reduced(),
    )
    params = _data_dict.keys()
    param_names = ["input_data"]

    def setup(self, input_data):
        self.adata = rsc.get.anndata_to_GPU(
            self._data_dict[input_data].copy(), copy=True
        )

    def time_umap(self, *_):
        rsc.tl.umap(self.adata)

    @track_peakmem
    def track_peakmem_umap(self, *_):
        rsc.tl.umap(self.adata)

    def time_diffmap(self, *_):
        rsc.tl.diffmap(self.adata)

    @track_peakmem
    def track_peakmem_diffmap(self, *_):
        rsc.tl.diffmap(self.adata)

    def time_leiden(self, *_):
        rsc.tl.leiden(self.adata)

    @track_peakmem
    def track_peakmem_leiden(self, *_):
        rsc.tl.leiden(self.adata)

    def time_embedding_denity(self, *_):
        rsc.tl.embedding_density(self.adata, basis="umap")

    @track_peakmem
    def track_peakmem_embedding_denity(self, *_):
        rsc.tl.embedding_density(self.adata, basis="umap")
