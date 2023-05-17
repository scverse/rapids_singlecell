import pytest
from rapids_singlecell.gr import spatial_autocorr
import pandas as pd
from anndata import read_h5ad
from pathlib import Path
import numpy as np

MORAN_I = "moranI"
GEARY_C = "gearyC"


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_consistency(mode):
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    spatial_autocorr(dummy_adata, mode=mode)
    df1 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=50)
    df2 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=50)

    idx_df = df1.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_I
    elif mode == "geary":
        UNS_KEY = GEARY_C

    assert UNS_KEY in dummy_adata.uns.keys()
    assert "pval_sim_fdr_bh" in df1
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df1.columns.shape == (9,)
    # test pval_norm same
    np.testing.assert_allclose(
        df1["pval_norm"].values, df2["pval_norm"].values, atol=1e-5, rtol=1e-5
    )
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
