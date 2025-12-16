from __future__ import annotations

import os

import pertpy as pt
import scanpy as sc

import rapids_singlecell as rsc

if __name__ == "__main__":
    adata = pt.data.adamson_2016_upr_epistasis()
    obs_key = "perturbation"

    # remove genes with 0 expression
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    # fill na obskeys
    # set categories first
    adata.obs[obs_key] = adata.obs[obs_key].cat.add_categories("control")
    adata.obs[obs_key] = adata.obs[obs_key].fillna("control")
    rsc.pp.pca(adata, n_comps=50)
    # save dir as
    # homedir/data/adamson_2016_upr_epistasis
    save_dir = os.path.join(os.path.expanduser("~"), "data")
    os.makedirs(save_dir, exist_ok=True)
    adata.write(os.path.join(save_dir, "adamson_2016_upr_epistasis_pca.h5ad"))
