import squidpy as sq

from pathlib import Path
import os
HOME = Path(os.path.expanduser("~"))
if __name__ == "__main__":
    (HOME / "data").mkdir(parents=True, exist_ok=True)
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    adata.write_h5ad(HOME / "data/visium_hne_adata.h5ad")