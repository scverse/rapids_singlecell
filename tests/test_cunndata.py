import cupy as cp
import cupyx.scipy.sparse as spg
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from rapids_singlecell.cunnData import cunnData
from scipy import sparse as sp


def test_create_with_dfs():
    X = cp.ones((6, 3))
    obs = pd.DataFrame({"cat_anno": pd.Categorical(["a", "a", "a", "a", "b", "a"])})
    obs_copy = obs.copy()
    adata = cunnData(X=X, obs=obs)
    assert obs.index.equals(obs_copy.index)
    assert obs.index.astype(str).equals(adata.obs.index)


def test_creation():
    cunnData(X=np.array([[1, 2], [3, 4]]))
    cunnData(X=np.array([[1, 2], [3, 4]]), obs={}, var={})
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cunnData(
        X=X,
        obs={"Obs": ["A", "B"]},
        var={"Feat": ["a", "b", "c"]},
        obsm={"X_pca": np.array([[1, 2], [3, 4]])},
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_dtype(dtype):
    cudata = cunnData(X=np.array([[1, 2], [3, 4]], dtype=dtype))
    assert cudata.X.dtype == np.float32


@pytest.mark.parametrize("mtype", ["csr", "csc", "coo", "array"])
def check_cpu_type(mtype):
    if mtype not in {"csr", "csc", "coo"}:
        X = sp.random(30, 20, density=0.2, format=mtype, dtype=None)
    else:
        X = np.random.randint(0, 10, size=(30, 20))
    cudata = cunnData(X=X)
    assert type(cudata.X) is spg.csr_matrix


def check_X_error_shape():
    X = spg.random(30, 20, density=0.2, format="csr", dtype=None)
    cudata = cunnData(X=X)
    with pytest.raises(ValueError):
        cudata.X = spg.random(30, 30, density=0.2, format="csr", dtype=None)


def check_var_error_shape():
    X = spg.random(30, 20, density=0.2, format="csr", dtype=None)
    cudata = cunnData(X=X)
    var = pd.DataFrame(index=range(cudata.shape[1] + 1))
    with pytest.raises(ValueError):
        cudata.var = var


def check_obs_error_shape():
    X = spg.random(30, 20, density=0.2, format="csr", dtype=None)
    cudata = cunnData(X=X)
    obs = pd.DataFrame(index=range(cudata.shape[0] + 1))
    with pytest.raises(ValueError):
        cudata.obs = obs


def test_slicing():
    cudata = cunnData(X=cp.array([[1, 2, 3], [4, 5, 6]]))

    assert cudata[0, :].X.get().tolist() == np.reshape([1, 2, 3], (1, 3)).tolist()
    assert cudata[:, 0].X.get().tolist() == np.reshape([1, 4], (2, 1)).tolist()

    assert cudata[:, [0, 1]].X.get().tolist() == [[1, 2], [4, 5]]
    assert cudata[:, np.array([0, 2])].X.tolist() == [[1, 3], [4, 6]]
    assert cudata[:, np.array([False, True, True])].X.tolist() == [
        [2, 3],
        [5, 6],
    ]
    assert cudata[:, 1:3].X.tolist() == [[2, 3], [5, 6]]

    assert cudata[0:2, :][:, 0:2].X.tolist() == [[1, 2], [4, 5]]
    assert cudata[0:1, :][:, 0:2].X.tolist() == np.reshape([1, 2], (1, 2)).tolist()
    assert cudata[0, :][:, 0].X.tolist() == np.reshape(1, (1, 1)).tolist()
    assert cudata[:, 0:2][0:2, :].X.tolist() == [[1, 2], [4, 5]]
    assert cudata[:, 0:2][0:1, :].X.tolist() == np.reshape([1, 2], (1, 2)).tolist()
    assert cudata[:, 0][0, :].X.tolist() == np.reshape(1, (1, 1)).tolist()


def test_slicing_sparse():
    cudata = cunnData(X=np.array([[1, 2, 3], [4, 5, 6]]))

    assert cudata[0, :].X.toarray().tolist() == np.reshape([1, 2, 3], (1, 3)).tolist()
    assert cudata[:, 0].X.toarray().tolist() == np.reshape([1, 4], (2, 1)).tolist()

    assert cudata[:, [0, 1]].X.toarray().tolist() == [[1, 2], [4, 5]]
    assert cudata[:, np.array([0, 2])].X.toarray().tolist() == [[1, 3], [4, 6]]
    assert cudata[:, np.array([False, True, True])].X.toarray().tolist() == [
        [2, 3],
        [5, 6],
    ]
    assert cudata[:, 1:3].X.toarray().tolist() == [[2, 3], [5, 6]]

    assert cudata[0:2, :][:, 0:2].X.toarray().tolist() == [[1, 2], [4, 5]]
    assert (
        cudata[0:1, :][:, 0:2].X.toarray().tolist()
        == np.reshape([1, 2], (1, 2)).tolist()
    )
    assert cudata[0, :][:, 0].X.toarray().tolist() == np.reshape(1, (1, 1)).tolist()
    assert cudata[:, 0:2][0:2, :].X.toarray().tolist() == [[1, 2], [4, 5]]
    assert (
        cudata[:, 0:2][0:1, :].X.toarray().tolist()
        == np.reshape([1, 2], (1, 2)).tolist()
    )
    assert cudata[:, 0][0, :].X.toarray().tolist() == np.reshape(1, (1, 1)).tolist()


def test_slicing_layer():
    cudata = cunnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    cudata.layers["counts"] = cp.array([[1, 2, 3], [4, 5, 6]])

    assert (
        cudata[0, :].layers["counts"].tolist() == np.reshape([1, 2, 3], (1, 3)).tolist()
    )
    assert cudata[:, 0].layers["counts"].tolist() == np.reshape([1, 4], (2, 1)).tolist()

    assert cudata[:, [0, 1]].layers["counts"].tolist() == [[1, 2], [4, 5]]
    assert cudata[:, np.array([0, 2])].layers["counts"].tolist() == [[1, 3], [4, 6]]
    assert cudata[:, np.array([False, True, True])].layers["counts"].tolist() == [
        [2, 3],
        [5, 6],
    ]
    assert cudata[:, 1:3].X.toarray().tolist() == [[2, 3], [5, 6]]

    assert cudata[0:2, :][:, 0:2].layers["counts"].tolist() == [[1, 2], [4, 5]]
    assert (
        cudata[0:1, :][:, 0:2].layers["counts"].tolist()
        == np.reshape([1, 2], (1, 2)).tolist()
    )
    assert (
        cudata[0, :][:, 0].layers["counts"].tolist() == np.reshape(1, (1, 1)).tolist()
    )
    assert cudata[:, 0:2][0:2, :].layers["counts"].tolist() == [[1, 2], [4, 5]]
    assert (
        cudata[:, 0:2][0:1, :].layers["counts"].tolist()
        == np.reshape([1, 2], (1, 2)).tolist()
    )
    assert (
        cudata[:, 0][0, :].layers["counts"].tolist() == np.reshape(1, (1, 1)).tolist()
    )


def test_slicing_varm_obsm():
    cudata = cunnData(X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    cudata.varm["test"] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cudata.obsm["test"] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert cudata[0, 0].varm["test"].tolist() == [[1, 2, 3]]
    assert cudata[0, 0].obsm["test"].tolist() == [[1, 2, 3]]
    assert cudata[:, 0].varm["test"].tolist() == [[1, 2, 3]]
    assert cudata[:, 0].obsm["test"].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert cudata[0, :].varm["test"].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert cudata[0, :].obsm["test"].tolist() == [[1, 2, 3]]
    assert cudata[:2, :2].varm["test"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert cudata[:2, :2].obsm["test"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_slicing_varm_obsm_pd():
    cudata = cunnData(X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    cudata.varm["test"] = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    cudata.obsm["test"] = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert cudata[0, 0].varm["test"].to_numpy().tolist() == [[1, 2, 3]]
    assert cudata[0, 0].obsm["test"].to_numpy().tolist() == [[1, 2, 3]]
    assert cudata[:, 0].varm["test"].to_numpy().tolist() == [[1, 2, 3]]
    assert cudata[:, 0].obsm["test"].to_numpy().tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert cudata[0, :].varm["test"].to_numpy().tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert cudata[0, :].obsm["test"].to_numpy().tolist() == [[1, 2, 3]]
    assert cudata[:2, :2].varm["test"].to_numpy().tolist() == [[1, 2, 3], [4, 5, 6]]
    assert cudata[:2, :2].obsm["test"].to_numpy().tolist() == [[1, 2, 3], [4, 5, 6]]


def test_wrong_demension_handling():
    cudata = cunnData(X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    with pytest.raises(ValueError):
        cudata.varm["test"] = np.array([[1, 2], [4, 5]])
    with pytest.raises(ValueError):
        cudata.obsm["test"] = np.array([[1, 2], [4, 5]])
    with pytest.raises(ValueError):
        cudata.layers["test"] = np.array([[1, 2], [4, 5]])


def test_set_X_layers():
    cudata = cunnData(X=cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    cudata.layers["test"] = cudata.X.copy()
    cudata.X = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 2
    cudata.layers["test"] = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 2
    assert (
        cudata.X.tolist() == (cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 2).tolist()
    )
    assert (
        cudata.layers["test"].tolist()
        == (cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * 2).tolist()
    )


def test_amgibuous_keys():
    """Tests that an error is raised if obs_vector or var_vector is ambiguous."""
    var_keys = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    obs_keys = [
        "Lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
    ]
    adata = AnnData(
        X=sp.random(len(obs_keys), len(var_keys), format="csr"),
        layers={"layer": sp.random(len(obs_keys), len(var_keys), format="csr")},
        obs=pd.DataFrame(
            np.random.randn(len(obs_keys), len(obs_keys) + len(var_keys)),
            index=obs_keys,
            columns=obs_keys + var_keys,
        ),
        var=pd.DataFrame(
            np.random.randn(len(var_keys), len(obs_keys) + len(var_keys)),
            index=var_keys,
            columns=var_keys + obs_keys,
        ),
    )

    cudata = cunnData(adata)

    for k in var_keys:
        # These are mostly to check that the test is working
        assert k in cudata.var_names
        assert k in cudata.obs.columns
        # Now the actual checks:
        with pytest.raises(ValueError, match=r".*var_names.*obs\.columns.*"):
            cudata.obs_vector(k)
        with pytest.raises(ValueError, match=r".*var_names.*obs\.columns.*"):
            cudata.obs_vector(k, layer="layer")

        # Should uniquely select column from in adata.var
        assert list(cudata.var[k]) == list(cudata.var_vector(k))
        assert list(cudata.var[k]) == list(cudata.var_vector(k, layer="layer"))

    for k in obs_keys:
        assert k in cudata.obs_names
        assert k in cudata.var.columns
        with pytest.raises(ValueError, match=r".*obs_names.*var\.columns"):
            cudata.var_vector(k)
        with pytest.raises(ValueError, match=r".*obs_names.*var\.columns"):
            cudata.var_vector(k, layer="layer")

        assert list(cudata.obs[k]) == list(cudata.obs_vector(k))
        assert list(cudata.obs[k]) == list(cudata.obs_vector(k, layer="layer"))
