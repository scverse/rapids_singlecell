import squidpy as sq
import warnings
warnings.filterwarnings("ignore")
import time
import anndata as ad
from pathlib import Path
import os

from utils.sepal_cpu import sepal
HOME = Path(os.path.expanduser("~"))
if __name__ == "__main__":
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    start_time = time.time()
    result = sepal(adata, max_neighs=6, genes=adata.var_names.values[:100], n_iter=4000, copy=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    result.sort_values(by="sepal_score", ascending=False, inplace=True)
    print(result.head(10))