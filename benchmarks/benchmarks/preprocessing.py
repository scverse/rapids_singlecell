"""
This module will benchmark preprocessing operations in Scanpy
API documentation: https://scanpy.readthedocs.io/en/stable/api/preprocessing.html
"""

from __future__ import annotations

import scanpy as sc

import rapids_singlecell as rsc

from .utils import track_peakmem


class PreprocessingSuite:
    _data_dict = dict(pbmc68k_reduced=sc.datasets.pbmc68k_reduced())
    params = _data_dict.keys()
    param_names = ["input_data"]

    def setup(self, input_data: str):
        self.adata = rsc.get.anndata_to_GPU(
            self._data_dict[input_data].copy(), copy=True
        )

    def time_calculate_qc_metrics(self, *_):
        self.adata.var["mt"] = self.adata.var_names.str.startswith("MT-")
        rsc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt"], log1p=False)

    @track_peakmem
    def track_peakmem_calculate_qc_metrics(self, *_):
        self.adata.var["mt"] = self.adata.var_names.str.startswith("MT-")
        rsc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt"], log1p=False)

    def time_filter_cells(self, *_):
        rsc.pp.filter_cells(self.adata, qc_var="n_counts", min_count=200)

    @track_peakmem
    def track_peakmem_filter_cells(self, *_):
        rsc.pp.filter_cells(self.adata, qc_var="n_counts", min_count=200)

    def time_filter_genes(self, *_):
        rsc.pp.filter_genes(self.adata, qc_var="n_counts", min_count=3)

    @track_peakmem
    def track_peakmem_filter_genes(self, *_):
        rsc.pp.filter_genes(self.adata, qc_var="n_counts", min_count=3)

    def time_normalize_total(self, *_):
        rsc.pp.normalize_total(self.adata, target_sum=1e4)

    @track_peakmem
    def track_peakmem_normalize_total(self, *_):
        rsc.pp.normalize_total(self.adata, target_sum=1e4)

    def time_log1p(self, *_):
        rsc.pp.log1p(self.adata)

    @track_peakmem
    def track_peakmem_time_log1p(self, *_):
        rsc.pp.log1p(self.adata)

    def time_pca(self, *_):
        rsc.pp.pca(self.adata)

    @track_peakmem
    def track_peakmem_pca(self, *_):
        rsc.pp.pca(self.adata)

    def time_highly_variable_genes(self, *_):
        rsc.pp.highly_variable_genes(
            self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5
        )

    @track_peakmem
    def track_peakmem_highly_variable_genes(self, *_):
        rsc.pp.highly_variable_genes(
            self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5
        )

    def time_regress_out(self, *_):
        rsc.pp.regress_out(self.adata, ["n_counts", "percent_mito"])

    @track_peakmem
    def track_peakmem_regress_out(self, *_):
        rsc.pp.regress_out(self.adata, ["n_counts", "percent_mito"])

    def time_scale(self, *_):
        rsc.pp.scale(self.adata, max_value=10)

    @track_peakmem
    def track_peakmem_scale(self, *_):
        rsc.pp.scale(self.adata, max_value=10)

    def time_neighbors(self, *_):
        rsc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=50)

    @track_peakmem
    def track_peakmem_neighbors(self, *_):
        rsc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=50)
