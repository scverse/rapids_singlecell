from __future__ import annotations

import os
import time

import anndata as ad
from pertpy.tools import Distance

if __name__ == "__main__":
    obs_key = "perturbation"

    # homedir/data/adamson_2016_upr_epistasis
    save_dir = os.path.join(
        os.path.expanduser("~"), "data", "adamson_2016_upr_epistasis_pca.h5ad"
    )
    adata = ad.read_h5ad(save_dir)
    dist = Distance(obsm_key="X_pca", metric="edistance")
    start_time = time.time()
    df = dist.pairwise(adata, groupby=obs_key)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
