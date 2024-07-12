import cupy as cp
import numpy as np
import pandas as pd
import cudf
from anndata import AnnData
from rapids_singlecell.dcg import run_mlm, run_wsum, run_ulm
from rapids_singlecell.decoupler_gpu._method_mlm import mlm
from rapids_singlecell.decoupler_gpu._method_wsum import wsum
from rapids_singlecell.decoupler_gpu._method_ulm import ulm, mat_cov, mat_cor, t_val
from rapids_singlecell.decoupler_gpu._pre import extract
import pytest
from scipy.sparse import csr_matrix
from numpy.testing import assert_almost_equal

# Test run_mlm
def test_mlm():
    m = csr_matrix(np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]], dtype=np.float32))
    net = np.array([[1., 0.], [2, 0.], [0., -3.], [0., 4.]], dtype=np.float32)
    act, pvl = mlm(m, net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0. <= pvl) * (pvl <= 1.))
    act, pvl = mlm(cp.array(m.toarray()), net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0. <= pvl) * (pvl <= 1.))


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
    m = csr_matrix(np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]], dtype=np.float32))
    net = np.array([[1., 0.], [2, 0.], [0., -3.], [0., 4.]], dtype=np.float32)
    est, norm, corr, pvl = wsum(m, net, 1000, 10000, 42, True)
    assert norm[0, 0] > 0
    assert norm[1, 0] > 0
    assert norm[2, 0] < 0
    assert norm[3, 0] < 0
    assert np.all((0. <= pvl) * (pvl <= 1.))
    est, norm, corr, pvl = wsum(cp.array(m.toarray()), net, 1000, 10000, 42, True)
    assert norm[0, 0] > 0
    assert norm[1, 0] > 0
    assert norm[2, 0] < 0
    assert norm[3, 0] < 0
    assert np.all((0. <= pvl) * (pvl <= 1.))


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
    m = csr_matrix(np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]], dtype=np.float32))
    net = np.array([[1., 0.], [2, 0.], [0., -3.], [0., 4.]], dtype=np.float32)
    act, pvl = ulm(m, net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0. <= pvl) * (pvl <= 1.))
    act, pvl = ulm(cp.array(m.toarray()), net)
    assert act[0, 0] > 0
    assert act[1, 0] > 0
    assert act[2, 0] < 0
    assert act[3, 0] < 0
    assert np.all((0. <= pvl) * (pvl <= 1.))

def test_run_ulm():
    m = np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 2], ['T2', 'G3', -3], ['T2', 'G4', 4]],
                    columns=['source', 'target', 'weight'])
    run_ulm(adata, net, verbose=True, use_raw=False, min_n=0)

def test_mat_cov():
    A = cp.array([
        [4, 5, 0, 1],
        [6, 3, 1, 0],
        [2, 3, 5, 5]
    ])

    b = cp.array([
        [1, 0],
        [1.7, 0],
        [0, 2.5],
        [0, 1]
    ])

    dc_cov = mat_cov(b, A.T)
    np_cov = cp.cov(b, A.T, rowvar=False)[:2, 2:].T
    cp.testing.assert_array_almost_equal(dc_cov, np_cov)


def test_mat_cor():
    A = cp.array([
        [4, 5, 0, 1],
        [6, 3, 1, 0],
        [2, 3, 5, 5]
    ])

    b = cp.array([
        [1, 0],
        [1.7, 0],
        [0, 2.5],
        [0, 1]
    ])

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
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
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
        extract('asdfg')
    with pytest.raises(AttributeError):
        extract(adata, use_raw=True)
