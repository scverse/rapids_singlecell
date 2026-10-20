from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy.sparse import csc_matrix as gpu_csc
from cupyx.scipy.sparse import csr_matrix as gpu_csr

import rapids_singlecell as rsc
from rapids_singlecell.get import X_to_CPU


def _synthetic_counts(
    rng: np.random.Generator,
    n_cells: int,
    n_guides: int,
    *,
    frac_pos: float,
    log2_mean: float,
    background: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n_cells, n_guides), dtype=np.float32)
    truth = np.zeros((n_cells, n_guides), dtype=bool)
    for g in range(n_guides):
        is_pos = rng.random(n_cells) < frac_pos
        X[is_pos, g] = np.clip(
            np.round(2 ** rng.normal(log2_mean, 0.6, is_pos.sum())), 1, None
        )
        X[~is_pos, g] = rng.poisson(background, (~is_pos).sum())
        truth[is_pos, g] = True
    return X, truth


def _adata(X: np.ndarray) -> AnnData:
    return AnnData(
        X=cp.asarray(X),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i}" for i in range(X.shape[1])]),
    )


@pytest.fixture
def guide_adata() -> tuple[AnnData, np.ndarray]:
    """800 cells by 12 guides with 12 % positive cells per guide."""
    X, truth = _synthetic_counts(
        np.random.default_rng(0), 800, 12, frac_pos=0.12, log2_mean=5.0, background=0.8
    )
    return _adata(X), truth


@pytest.fixture
def guide_adata_sparse(
    guide_adata: tuple[AnnData, np.ndarray],
) -> tuple[AnnData, np.ndarray]:
    adata, truth = guide_adata
    adata = adata.copy()
    adata.X = gpu_csr(adata.X)
    return adata, truth


@pytest.fixture
def tiny_adata() -> AnnData:
    return _adata(np.array([[6, 0, 2], [1, 5, 0], [0, 1, 7]], dtype=np.float32))


def _dense(X) -> np.ndarray:
    X = X_to_CPU(X)
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_assign_by_threshold(tiny_adata: AnnData, storage: str) -> None:
    if storage != "dense":
        tiny_adata.X = (gpu_csr if storage == "csr" else gpu_csc)(tiny_adata.X)
    rsc.ptg.GuideAssignment().assign_by_threshold(tiny_adata, assignment_threshold=5)

    result = tiny_adata.layers["assigned_guides"]
    assert type(result) is type(tiny_adata.X)
    if storage == "dense":
        assert result.dtype == cp.int8
    else:
        assert result.nnz == 3
    np.testing.assert_array_equal(
        _dense(result), (_dense(tiny_adata.X) >= 5).astype(np.int8)
    )


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_assign_to_max_guide(tiny_adata: AnnData, storage: str) -> None:
    if storage != "dense":
        tiny_adata.X = (gpu_csr if storage == "csr" else gpu_csc)(tiny_adata.X)
    rsc.ptg.GuideAssignment().assign_to_max_guide(tiny_adata, assignment_threshold=6)
    assert tuple(tiny_adata.obs["assigned_guide"]) == ("guide_0", "Negative", "guide_2")


@pytest.mark.parametrize("fixture", ["guide_adata", "guide_adata_sparse"])
def test_mixture_model_recovers_known_positives(request, fixture: str) -> None:
    adata, truth = request.getfixturevalue(fixture)
    rsc.ptg.GuideAssignment().assign_mixture_model(adata)

    thresholds = np.nan_to_num(adata.var["threshold"].to_numpy(), nan=np.inf)
    pred = _dense(adata.X) >= thresholds[None, :]
    assert (pred == truth).mean() > 0.98
    assert (pred & truth).sum() / truth.sum() > 0.9
    assert (pred & ~truth).sum() <= 0.005 * truth.size

    layer = adata.layers["assigned_guides"]
    assert type(layer) is type(adata.X)
    np.testing.assert_array_equal(_dense(layer), pred.astype(np.int8))
    for col in (
        "poisson_rate",
        "gaussian_mean",
        "gaussian_std",
        "mix_probs_0",
        "mix_probs_1",
        "threshold",
        "final_loss",
    ):
        assert np.isfinite(adata.var[col]).all()
    assert (adata.var["gaussian_mean"] > adata.var["poisson_rate"]).all()


def test_mixture_model_dense_matches_sparse(
    guide_adata: tuple[AnnData, np.ndarray],
    guide_adata_sparse: tuple[AnnData, np.ndarray],
) -> None:
    dense, _ = guide_adata
    sparse, _ = guide_adata_sparse
    rsc.ptg.GuideAssignment().assign_mixture_model(dense)
    rsc.ptg.GuideAssignment().assign_mixture_model(sparse)

    assert list(dense.obs["assigned_guide"]) == list(sparse.obs["assigned_guide"])
    np.testing.assert_array_equal(dense.var["threshold"], sparse.var["threshold"])


def test_mixture_model_high_moi_keeps_component_order() -> None:
    """With half of the cells positive, the Gaussian must stay the on-target component."""
    X, truth = _synthetic_counts(
        np.random.default_rng(0), 2000, 3, frac_pos=0.5, log2_mean=6.0, background=1.0
    )
    adata = _adata(X)
    rsc.ptg.GuideAssignment().assign_mixture_model(adata)

    thresholds = adata.var["threshold"].to_numpy()
    assert np.all(thresholds > 1)
    assert (adata.var["gaussian_mean"] > adata.var["poisson_rate"]).all()
    assert ((X >= thresholds[None, :]) == truth).mean() > 0.99


def test_mixture_model_multiple_guides(guide_adata: tuple[AnnData, np.ndarray]) -> None:
    adata, _ = guide_adata
    rsc.ptg.GuideAssignment().assign_mixture_model(
        adata, max_assignments_per_cell=1, multiple_grna_assigned_key="too_many"
    )
    n_assigned = _dense(adata.layers["assigned_guides"]).sum(axis=1)
    assigned = adata.obs["assigned_guide"].astype(str).to_numpy()
    assert (n_assigned > 1).any()
    assert (assigned[n_assigned == 0] == "negative").all()
    assert (assigned[n_assigned > 1] == "too_many").all()

    rsc.ptg.GuideAssignment().assign_mixture_model(adata, output_layer=None)
    joined = adata.obs["assigned_guide"].astype(str).to_numpy()[n_assigned == 2]
    assert all(len(label.split("+")) == 2 for label in joined)


def test_mixture_model_only_return_results(
    guide_adata: tuple[AnnData, np.ndarray],
) -> None:
    adata, _ = guide_adata
    result = rsc.ptg.GuideAssignment().assign_mixture_model(
        adata, only_return_results=True
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (adata.n_obs,)
    assert "assigned_guide" not in adata.obs
    assert "threshold" not in adata.var
    assert "assigned_guides" not in adata.layers


def test_mixture_model_layer_is_honored(
    guide_adata: tuple[AnnData, np.ndarray],
) -> None:
    adata, _ = guide_adata
    adata.layers["raw_guides"] = gpu_csc(adata.X)
    adata.X = cp.zeros_like(adata.X)

    ga = rsc.ptg.GuideAssignment()
    with pytest.warns(UserWarning, match="Skipping 12 guides"):
        x_result = ga.assign_mixture_model(adata, only_return_results=True)
    layer_result = ga.assign_mixture_model(
        adata, layer="raw_guides", only_return_results=True
    )

    assert (x_result == "negative").all()
    assert np.count_nonzero(layer_result != "negative") > 0


def test_mixture_model_skips_low_count_guides() -> None:
    X = np.zeros((200, 4), dtype=np.float32)
    X[0, 1] = 1
    X[:50, 2] = 1
    rng = np.random.default_rng(0)
    is_pos = rng.random(200) < 0.15
    X[is_pos, 3] = np.round(2 ** rng.normal(5.0, 0.6, is_pos.sum()))
    X[~is_pos, 3] = rng.poisson(0.8, (~is_pos).sum())
    adata = _adata(X)

    with pytest.warns(
        UserWarning, match="Skipping 3 guides .*: guide_0, guide_1, guide_2"
    ):
        rsc.ptg.GuideAssignment().assign_mixture_model(adata)
    assert adata.var["threshold"].isna().tolist() == [True, True, True, False]


def test_mixture_model_rejects_negative_values_and_unknown_model(
    tiny_adata: AnnData,
) -> None:
    with pytest.raises(ValueError, match="Model not implemented"):
        rsc.ptg.GuideAssignment().assign_mixture_model(tiny_adata, model="not_a_model")
    tiny_adata.X[0, 0] = -1
    with pytest.raises(ValueError, match="negative values"):
        rsc.ptg.GuideAssignment().assign_mixture_model(tiny_adata)
