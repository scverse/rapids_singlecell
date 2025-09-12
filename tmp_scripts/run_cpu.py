import warnings
warnings.filterwarnings("ignore")
import squidpy as sq
import time
import anndata as ad
from pathlib import Path
import os
from argparse import ArgumentParser
from utils.sepal_cpu import sepal
HOME = Path(os.path.expanduser("~"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    start_time = time.time()
    result = sepal(adata, max_neighs=6, genes=adata.var_names.values[:10], n_iter=4000, copy=True, debug=args.debug)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    result.sort_values(by="sepal_score", ascending=False, inplace=True)
    print(result.head(10))