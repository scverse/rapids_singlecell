"""
This module will benchmark tool operations in Scanpy
API documentation: https://scanpy.readthedocs.io/en/stable/api/tools.html
"""

from __future__ import annotations

from itertools import product

import anndata as ad

import rapids_singlecell as rsc

import pathlib

class ToolsSuite:
    _data_dict = dict(
        visium_sge=ad.read_h5ad("/p/project/training2406/team_scverse/gold2/rapids_singlecell/benchmarks/data/paul15.h5ad"),
    )
    params = _data_dict.keys()
    param_names = ["input_data"]

    def setup(self, input_data):
        self.adata = rsc.get.anndata_to_GPU(self._data_dict[input_data].copy(), copy=True)

    def time_ligrec(self, *_):
        gene_ids = self.adata.var.index
        interactions = tuple(product(gene_ids[:5], gene_ids[:5]))
        rsc.gr.ligrec(
            self.adata,
            "leiden",
            interactions=interactions,
            n_perms=5,
            use_raw=False,
        )

    def peakmem_ligrec(self, *_):
        gene_ids = self.adata.var.index
        interactions = tuple(product(gene_ids[:5], gene_ids[:5]))
        rsc.gr.ligrec(
            self.adata,
            "leiden",
            interactions=interactions,
            n_perms=5,
            use_raw=False,
        )

    def time_autocorr_moran(self, *_):
        rsc.gr.spatial_autocorr(self.adata, mode="moran")

    def peakmem_autocorr_moran(self, *_):
        rsc.gr.spatial_autocorr(self.adata, mode="moran")

    def time_autocorr_geary(self, *_):
        rsc.gr.spatial_autocorr(self.adata, mode="geary")

    def peakmem_autocorr_geary(self, *_):
        rsc.gr.spatial_autocorr(self.adata, mode="geary")
