from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as sts

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu._helper._pv import fdr_bh_axis1


@pytest.mark.parametrize("if_gpu", [True, False])
def test_func_mlm(
    adata,
    net,
    if_gpu,
):
    dc.mlm(data=adata, net=net, tmin=3)
    dc_pv = adata.obsm["padj_mlm"]
    adj = fdr_bh_axis1(dc_pv.values, if_gpu=if_gpu)
    np.testing.assert_allclose(
        adj, sts.false_discovery_control(dc_pv.values, axis=1, method="bh")
    )
