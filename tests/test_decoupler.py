from __future__ import annotations

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix

from rapids_singlecell.dcg import run_aucell, run_mlm, run_ulm, run_wsum
from rapids_singlecell.decoupler_gpu._method_aucell import aucell
from rapids_singlecell.decoupler_gpu._method_mlm import mlm
from rapids_singlecell.decoupler_gpu._method_ulm import mat_cor, mat_cov, t_val, ulm
from rapids_singlecell.decoupler_gpu._method_wsum import wsum
from rapids_singlecell.decoupler_gpu._pre import extract


# Test run_aucell
def test_aucell():
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
    net = pd.Series(
        [np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)],
        index=["T1", "T2"],
    )
    n_up = np.array([4], dtype=np.int64)[0]
    aucell(m, net, n_up, False)

    act = aucell(m, net, n_up, False)
    assert act[0, 0] > 0.7
    assert act[1, 0] > 0.7
    assert act[2, 0] < 0.7
    assert act[3, 0] < 0.7
    assert np.all((0.0 <= act) * (act <= 1.0))
    act = aucell(m.toarray(), net, n_up, False)
    assert act[0, 0] > 0.7
    assert act[1, 0] > 0.7
    assert act[2, 0] < 0.7
    assert act[3, 0] < 0.7
    assert np.all((0.0 <= act) * (act <= 1.0))


def test_run_aucell():
    m = np.array([[7.0, 1.0, 1.0], [4.0, 2.0, 1.0], [1.0, 2.0, 5.0], [1.0, 1.0, 6.0]])
    r = np.array(["S1", "S2", "S3", "S4"])
    c = np.array(["G1", "G2", "G3"])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame(
        [["T1", "G2"], ["T1", "G4"], ["T2", "G3"], ["T2", "G1"]],
        columns=["source", "target"],
    )
    run_aucell(adata, net, n_up=2, min_n=0, verbose=True, use_raw=False)
    with pytest.raises(ValueError):
        run_aucell(adata, net, n_up=-3, min_n=0, verbose=True, use_raw=False)


# Test run_mlm
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
    act, pvl = mlm(m, net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))
    act, pvl = mlm(cp.array(m.toarray()), net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))


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


# Test run_wsum
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
    est, norm, corr, pvl = wsum(m, net, 1000, 10000, 42, True)
    assert norm[0, 0] > 0
    assert norm[1, 0] > 0
    assert norm[2, 0] < 0
    assert norm[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))
    est, norm, corr, pvl = wsum(cp.array(m.toarray()), net, 1000, 10000, 42, True)
    assert norm[0, 0] > 0
    assert norm[1, 0] > 0
    assert norm[2, 0] < 0
    assert norm[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))


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


# Test run_ulm
def test_ulm():
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
    act, pvl = ulm(m, net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))
    act, pvl = ulm(cp.array(m.toarray()), net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))


def test_run_ulm():
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
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame(
        [["T1", "G1", 1], ["T1", "G2", 2], ["T2", "G3", -3], ["T2", "G4", 4]],
        columns=["source", "target", "weight"],
    )
    run_ulm(adata, net, verbose=True, use_raw=False, min_n=0)


def test_mat_cov():
    A = cp.array([[4, 5, 0, 1], [6, 3, 1, 0], [2, 3, 5, 5]])

    b = cp.array([[1, 0], [1.7, 0], [0, 2.5], [0, 1]])

    dc_cov = mat_cov(b, A.T)
    np_cov = cp.cov(b, A.T, rowvar=False)[:2, 2:].T
    cp.testing.assert_array_almost_equal(dc_cov, np_cov)


def test_mat_cor():
    A = cp.array([[4, 5, 0, 1], [6, 3, 1, 0], [2, 3, 5, 5]])

    b = cp.array([[1, 0], [1.7, 0], [0, 2.5], [0, 1]])

    dc_cor = mat_cor(b, A.T)
    np_cor = cp.corrcoef(b, A.T, rowvar=False)[:2, 2:].T
    cp.testing.assert_array_almost_equal(dc_cor, np_cor)
    assert cp.all((dc_cor <= 1) * (dc_cor >= -1))


def test_t_val():
    t = t_val(r=0.4, df=28)
    assert_almost_equal(2.30940108, t)

    t = t_val(r=0.99, df=3)
    assert_almost_equal(12.15540081, t)

    t = t_val(r=-0.05, df=99)
    assert_almost_equal(-0.49811675, t)


def test_extract():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(["S1", "S2", "S3"])
    c = np.array(["G1", "G2", "G3"])
    df = pd.DataFrame(m, index=r, columns=c)
    cdf = cudf.DataFrame(m, index=r, columns=c)

    adata = AnnData(df.astype(np.float32))
    adata_raw = adata.copy()
    adata_raw.raw = adata_raw
    extract([m, r, c])
    extract(df)
    extract(cdf)
    extract(adata, use_raw=False)
    extract(adata_raw, use_raw=True)
    with pytest.raises(ValueError):
        extract("asdfg")
    with pytest.raises(AttributeError):
        extract(adata, use_raw=True)
