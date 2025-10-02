from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import anndata as ad
from utils.sepal_cpu import sepal

HOME = Path(os.path.expanduser("~"))

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    start_time = time.time()
    genes = ["Gm29570"]
    genes = adata.var_names.values[:100]
    # sc.pp.normalize_total(adata)
    result = sepal(
        adata, max_neighs=6, genes=genes, n_iter=30000, copy=True, 
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    result.sort_values(by="sepal_score", ascending=False, inplace=True)
    print(result.head(10))
