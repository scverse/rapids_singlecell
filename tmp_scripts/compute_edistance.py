from __future__ import annotations

import os
import time

import anndata as ad
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

import rapids_singlecell as rsc
from rapids_singlecell.ptg import Distance

rmm.reinitialize(
    managed_memory=False,  # Allows oversubscription
    pool_allocator=True,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)


if __name__ == "__main__":
    obs_key = "perturbation"

    # homedir/data/adamson_2016_upr_epistasis
    save_dir = os.path.join(
        os.path.expanduser("~"), "data", "adamson_2016_upr_epistasis_pca.h5ad"
    )
    adata = ad.read_h5ad(save_dir)
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    dist = Distance(obsm_key="X_pca", metric="edistance")
    start_time = time.time()
    df = dist.pairwise(adata, groupby=obs_key)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
