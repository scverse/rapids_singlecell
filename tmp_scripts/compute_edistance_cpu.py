from __future__ import annotations

import os
import time
from argparse import ArgumentParser
from pathlib import Path

import anndata as ad
from pertpy.tools import Distance

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    obs_key = "perturbation"
    bootstrap = args.bootstrap

    # homedir/data/adamson_2016_upr_epistasis
    save_dir = os.path.join(
        os.path.expanduser("~"),
        "data",
    )
    adata = ad.read_h5ad(Path(save_dir) / "adamson_2016_upr_epistasis_pca.h5ad")
    dist = Distance(obsm_key="X_pca", metric="edistance")
    start_time = time.time()
    if bootstrap:
        df, df_var = dist.pairwise(
            adata, groupby=obs_key, bootstrap=True, n_bootstrap=100
        )
    else:
        df = dist.pairwise(adata, groupby=obs_key)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    if bootstrap:
        df_var.to_csv(Path(save_dir) / "df_cpu_bootstrap_var.csv")
        df.to_csv(Path(save_dir) / "df_cpu_bootstrap.csv")
    else:
        df.to_csv(Path(save_dir) / "df_cpu.csv")
