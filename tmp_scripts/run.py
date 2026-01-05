from __future__ import annotations

import os
import time
from argparse import ArgumentParser
from pathlib import Path

import anndata as ad
from pertpy.tools import Distance

import rapids_singlecell as rsc
from rapids_singlecell.pertpy_gpu._edistance import pertpy_edistance

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    obs_key = "perturbation"
    bootstrap = args.bootstrap
    gpu = args.gpu
    # homedir/data/adamson_2016_upr_epistasis
    save_dir = os.path.join(
        os.path.expanduser("~"),
        "data",
    )
    adata = ad.read_h5ad(Path(save_dir) / "adamson_2016_upr_epistasis_pca.h5ad")
    if gpu:
        rsc.get.anndata_to_GPU(adata, convert_all=True)
    else:
        dist = Distance(obsm_key="X_pca", metric="edistance")
    start_time = time.time()
    if gpu:
        res = pertpy_edistance(
            adata,
            groupby=obs_key,
            obsm_key="X_pca",
            bootstrap=bootstrap,
            n_bootstrap=100,
        )
        df_gpu = res.distances
        df_gpu_var = res.distances_var
    else:
        if bootstrap:
            df, df_var = dist.pairwise(
                adata, groupby=obs_key, bootstrap=bootstrap, n_bootstrap=100
            )
        else:
            df = dist.pairwise(adata, groupby=obs_key)
            df_var = None
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
