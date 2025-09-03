from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps

import rapids_singlecell.decoupler_gpu as dc

"""
gs <- list(
    T1=c('G01', 'G02', 'G03'),
    T2=c('G04', 'G06', 'G07', 'G08'),
    T3=c('G06', 'G07', 'G08'),
    T4=c('G05', 'G10', 'G11', 'G09'),
    T5=c('G09', 'G10', 'G11')
)
mat <- matrix(c(
    0.879, 8.941, 1.951, 8.75, 0.128, 2.959, 2.369, 9.04, 0.853, 3.127, 0.017, 2.859, 0.316, 2.066, 2.05, 8.305, 0.778, 2.468, 1.302, 2.878,
    2.142, 8.155, 0.428, 9.223, 0.532, 2.84, 2.114, 8.681, 0.431, 2.814, 0.413, 3.129, 0.365, 2.512, 2.651, 8.185, 0.406, 2.616, 0.352, 2.824,
    1.729, 0.637, 8.341, 0.74, 8.084, 2.397, 3.093, 0.635, 1.682, 3.351, 1.28, 2.203, 8.556, 2.255, 3.303, 1.25, 1.359, 2.012, 9.784, 2.06,
    0.746, 0.894, 8.011, 1.798, 8.044, 3.059, 2.996, 0.08, 0.151, 2.391, 1.082, 2.123, 8.203, 2.511, 2.039, 0.051, 1.25, 3.787, 8.249, 3.026
), nrow=4, byrow=TRUE)
colnames(mat) <- c('G11', 'G04', 'G05', 'G03', 'G07', 'G18', 'G17', 'G02', 'G10',
       'G14', 'G09', 'G16', 'G08', 'G13', 'G20', 'G01', 'G12', 'G15',
       'G06', 'G19')
rownames(mat) <- c("S01", "S02", "S29", "S30")
rnks <- AUCell::AUCell_buildRankings(t(mat), plotStats=FALSE)
t(AUCell::AUCell_calcAUC(gs, rnks, aucMaxRank=3)@assays@data$AUC)
"""


def test_func_aucell(
    mat,
    idxmat,
):
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    obs = np.array(["S01", "S02", "S29", "S30"])
    X = np.vstack((X[:2, :], X[-2:, :]))
    X = sps.csr_matrix(X)
    ac_es = pd.DataFrame(
        data=np.array(
            [
                [0.6666667, 0.3333333, 0, 0, 0],
                [1.0000000, 0.0000000, 0, 0, 0],
                [0.0000000, 1.0000000, 1, 0, 0],
                [0.0000000, 1.0000000, 1, 0, 0],
            ]
        ),
        columns=["T1", "T2", "T3", "T4", "T5"],
        index=obs,
    )
    X_cp = cp.array(X.toarray())
    cnct_cp = cp.array(cnct, dtype=cp.int32)
    starts_cp = cp.array(starts, dtype=cp.int32)
    offsets_cp = cp.array(offsets, dtype=cp.int32)
    dc_es, _ = dc._method_aucell._func_aucell(
        mat=X_cp,
        cnct=cnct_cp,
        starts=starts_cp,
        offsets=offsets_cp,
        n_up=3,
    )
    assert np.isclose(dc_es, ac_es.values).all()
