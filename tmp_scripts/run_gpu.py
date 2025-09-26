from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import os
import time
from argparse import ArgumentParser
from pathlib import Path

import anndata as ad
from utils.sepal_gpu import sepal_gpu

import rapids_singlecell as rsc

HOME = Path(os.path.expanduser("~"))
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    # sc.pp.normalize_total(adata)
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    adata.obsp["spatial_connectivities"] = rsc.get.X_to_GPU(
        adata.obsp["spatial_connectivities"]
    )
    adata.obsm["spatial"] = rsc.get.X_to_GPU(adata.obsm["spatial"])
    start_time = time.time()
    genes = adata.var_names.values[:1000]
    # genes = ["Gm29570"]
    result = sepal_gpu(
        adata, max_neighs=6, genes=genes, n_iter=30000, copy=True, debug=args.debug
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    result.sort_values(by="sepal_score", ascending=False, inplace=True)
    print(result.head(10))
