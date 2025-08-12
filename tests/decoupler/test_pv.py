from __future__ import annotations

import numpy as np
import scipy.stats as sts

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu._helper._pv import fdr_bh_axis1


def test_func_mlm(
    adata,
    net,
):
    dc.mlm(data=adata, net=net, tmin=3)
    dc_pv = adata.obsm["padj_mlm"]
    adj = fdr_bh_axis1(dc_pv.values)
    np.testing.assert_allclose(
        adj, sts.false_discovery_control(dc_pv.values, axis=1, method="bh")
    )
