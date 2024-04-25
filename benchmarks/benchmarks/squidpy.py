"""
This module will benchmark tool operations in Scanpy
API documentation: https://scanpy.readthedocs.io/en/stable/api/tools.html
"""

from __future__ import annotations

from itertools import product

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
        self.cpu_adata = self._data_dict[input_data].copy()
        self.gpu_adata = rsc.get.anndata_to_GPU(self.cpu_adata, copy=True)

    def time_ligrec(self, *_):
        gene_ids = self.cpu_adata.var.index
        interactions = tuple(product(gene_ids[:5], gene_ids[:5]))
        rsc.gr.ligrec(
            self.cpu_adata,
            "louvain",
            interactions=interactions,
            n_perms=5,
            use_raw=False,
        )

    @track_peakmem
    def track_peakmem_ligrec(self, *_):
        gene_ids = self.cpu_adata.var.index
        interactions = tuple(product(gene_ids[:5], gene_ids[:5]))
        rsc.gr.ligrec(
            self.cpu_adata,
            "louvain",
            interactions=interactions,
            n_perms=5,
            use_raw=False,
        )


    def time_autocorr_moran(self, *_):
        rsc.gr.spatial_autocorr(self.gpu_adata, mode="moran", connectivity_key="connectivities")

    @track_peakmem
    def track_peakmem_autocorr_moran(self, *_):
        rsc.gr.spatial_autocorr(self.gpu_adata, mode="moran", connectivity_key="connectivities")

    def time_autocorr_geary(self, *_):
        rsc.gr.spatial_autocorr(self.gpu_adata, mode="geary", connectivity_key="connectivities")

    @track_peakmem
    def track_peakmem_autocorr_geary(self, *_):
        rsc.gr.spatial_autocorr(self.gpu_adata, mode="geary", connectivity_key="connectivities")

