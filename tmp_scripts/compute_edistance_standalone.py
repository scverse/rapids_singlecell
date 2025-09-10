from __future__ import annotations

import os
import time

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

import rapids_singlecell as rsc
from rapids_singlecell.pertpy_gpu._distances_standalone import pairwise_edistance_gpu

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
    save_dir_df = os.path.join(os.path.expanduser("~"), "data", "df_cpu_float64.csv")
    adata = ad.read_h5ad(save_dir)
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    start_time = time.time()
    df_gpu = pairwise_edistance_gpu(adata, groupby=obs_key, obsm_key="X_pca")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # print("CPU time")
    # dist = Distance(obsm_key="X_pca", metric="edistance")
    # start_time = time.time()
    # df_cpu = dist.pairwise(adata, groupby=obs_key)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    df = pd.read_csv(save_dir_df, index_col=0)

    is_not_close = []
    groups = adata.obs[obs_key].unique()
    k = len(groups)
    atol = 1e-8
    for idx1 in range(k):
        for idx2 in range(idx1 + 1, k):
            group_x = groups[idx1]
            group_y = groups[idx2]
            if group_x == group_y:
                assert df_gpu.loc[group_x, group_y] == 0
            else:
                if not np.isclose(
                    df_gpu.loc[group_x, group_y], df.loc[group_x, group_y], atol=atol
                ):
                    is_not_close.append(
                        ((group_x, group_y), df_gpu.loc[group_x, group_y], df.loc[group_x, group_y], np.abs(df_gpu.loc[group_x, group_y] - df.loc[group_x, group_y]))
                    )
                    print(f"Group df_gpu: {df_gpu.loc[group_x, group_y]}, Group df: {df.loc[group_x, group_y]}, idx: ({idx1}, {idx2})")
    
    print("Out of", int(k * (k - 1) / 2), "pairs,", len(is_not_close), "pairs are not close with atol=", atol)
    # print(df.equals(df_gpu))
    # print(df)
    # print(df_gpu)
