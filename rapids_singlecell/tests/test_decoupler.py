import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from rapids_singlecell.dcg import run_mlm, run_wsum
from rapids_singlecell.decoupler_gpu._method_mlm import mlm
from rapids_singlecell.decoupler_gpu._method_wsum import wsum


def test_mlm():
    m = csr_matrix(
        np.array(
            [
                [7.0, 1.0, 1.0, 1.0],
                [4.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 5.0, 1.0],
                [1.0, 1.0, 6.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    net = np.array([[1.0, 0.0], [2, 0.0], [0.0, -3.0], [0.0, 4.0]], dtype=np.float32)
    res = mlm(m, net)


def test_run_mlm():
    m = np.array(
        [
            [7.0, 1.0, 1.0, 1.0],
            [4.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 1.0],
            [1.0, 1.0, 6.0, 2.0],
        ]
    )
    r = np.array(["S1", "S2", "S3", "S4"])
    c = np.array(["G1", "G2", "G3", "G4"])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame(
        [["T1", "G1", 1], ["T1", "G2", 2], ["T2", "G3", -3], ["T2", "G4", 4]],
        columns=["source", "target", "weight"],
    )
    run_mlm(adata, net, verbose=False, use_raw=False, min_n=0)


def test_wsum():
    m = csr_matrix(
        np.array(
            [
                [7.0, 1.0, 1.0, 1.0],
                [4.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 5.0, 1.0],
                [1.0, 1.0, 6.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    net = np.array([[1.0, 0.0], [2, 0.0], [0.0, -3.0], [0.0, 4.0]], dtype=np.float32)
    res = wsum(m, net, 2, 10000, 42, True)


def test_run_wsum():
    m = np.array(
        [
            [7.0, 1.0, 1.0, 1.0],
            [4.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 1.0],
            [1.0, 1.0, 6.0, 2.0],
        ]
    )
    r = np.array(["S1", "S2", "S3", "S4"])
    c = np.array(["G1", "G2", "G3", "G4"])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame(
        [["T1", "G1", 1], ["T1", "G2", 2], ["T2", "G3", -3], ["T2", "G4", 4]],
        columns=["source", "target", "weight"],
    )
    run_wsum(adata, net, verbose=False, use_raw=False, min_n=0, times=2)
