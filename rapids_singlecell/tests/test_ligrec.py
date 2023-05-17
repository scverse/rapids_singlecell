import sys
from itertools import product
from time import time
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData, read_h5ad
from pandas.testing import assert_frame_equal
from scanpy import settings as s
from scanpy.datasets import blobs
from itertools import product

from rapids_singlecell.gr import ligrec
from pathlib import Path

_CK = "leiden"
Interactions_t = Tuple[Sequence[str], Sequence[str]]
Complexes_t = Sequence[Tuple[str, str]]


@pytest.fixture()
def adata() -> AnnData:
    file = Path(__file__).parent / Path("_data/test_data.h5ad")
    adata = read_h5ad(file)
    adata.raw = adata.copy()
    return adata


@pytest.fixture()
def interactions(adata: AnnData) -> Tuple[Sequence[str], Sequence[str]]:
    return tuple(product(adata.raw.var_names[:5], adata.raw.var_names[:5]))


class TestInvalidBehavior:
    def test_not_adata(self, interactions: Interactions_t):
        with pytest.raises(
            TypeError, match=r"Expected `adata` to be of type `anndata.AnnData`"
        ):
            ligrec(None, _CK, interactions=interactions)

    def test_adata_no_raw(self, adata: AnnData, interactions: Interactions_t):
        del adata.raw
        with pytest.raises(AttributeError, match=r"No `.raw` attribute"):
            ligrec(adata, _CK, use_raw=True, interactions=interactions)

    def test_raw_has_different_n_obs(
        self, adata: AnnData, interactions: Interactions_t
    ):
        adata.raw = sc.datasets.blobs(n_observations=adata.n_obs + 1)
        # raise below happend with anndata < 0.9
        # with pytest.raises(ValueError, match=rf"Expected `{adata.n_obs}` cells in `.raw`"):
        with pytest.raises(
            ValueError,
            match=rf"Index length mismatch: {adata.n_obs} vs. {adata.n_obs + 1}",
        ):
            ligrec(adata, _CK, interactions=interactions)

    def test_invalid_cluster_key(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(KeyError, match=r"Cluster key `foobar` not found"):
            ligrec(adata, cluster_key="foobar", interactions=interactions)

    def test_cluster_key_is_not_categorical(
        self, adata: AnnData, interactions: Interactions_t
    ):
        adata.obs[_CK] = adata.obs[_CK].astype("string")
        with pytest.raises(
            TypeError, match=rf"Expected `adata.obs\[{_CK!r}\]` to be `categorical`"
        ):
            ligrec(adata, _CK, interactions=interactions)

    def test_only_1_cluster(self, adata: AnnData, interactions: Interactions_t):
        adata.obs["foo"] = 1
        adata.obs["foo"] = adata.obs["foo"].astype("category")
        with pytest.raises(
            ValueError, match=rf"Expected at least `2` clusters, found `1`."
        ):
            ligrec(adata, "foo", interactions=interactions)

    def test_invalid_complex_policy(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(
            ValueError, match=rf"Invalid option `'foobar'` for `ComplexPolicy`."
        ):
            ligrec(adata, _CK, interactions=interactions, complex_policy="foobar")

    def test_invalid_fdr_axis(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(
            ValueError, match=r"Invalid option `foobar` for `CorrAxis`."
        ):
            ligrec(
                adata,
                _CK,
                interactions=interactions,
                corr_axis="foobar",
                corr_method="fdr_bh",
            )

    def test_too_few_permutations(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Expected `n_perms` to be positive"):
            ligrec(adata, _CK, interactions=interactions, n_perms=0)

    def test_invalid_interactions_type(self, adata: AnnData):
        with pytest.raises(TypeError, match=r"Expected either a `pandas.DataFrame`"):
            ligrec(adata, _CK, interactions=42)

    def test_invalid_interactions_dict(self, adata: AnnData):
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions={"foo": ["foo"], "target": ["bar"]})
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions={"source": ["foo"], "bar": ["bar"]})

    def test_invalid_interactions_dataframe(
        self, adata: AnnData, interactions: Interactions_t
    ):
        df = pd.DataFrame(interactions, columns=["foo", "target"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions=df)

        df = pd.DataFrame(interactions, columns=["source", "bar"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions=df)

    def test_interactions_invalid_sequence(
        self, adata: AnnData, interactions: Interactions_t
    ):
        interactions += ("foo", "bar", "bar")  # type: ignore
        with pytest.raises(
            ValueError, match=r"Not all interactions are of length `2`."
        ):
            ligrec(adata, _CK, interactions=interactions)

    def test_interactions_only_invalid_names(self, adata: AnnData):
        with pytest.raises(ValueError, match=r"After filtering by genes"):
            ligrec(adata, _CK, interactions=["foo", "bar", "baz"])

    def test_invalid_clusters(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Invalid cluster `'foo'`."):
            ligrec(adata, _CK, interactions=interactions, clusters=["foo"])

    def test_invalid_clusters_mix(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(
            ValueError, match=r"Expected a `tuple` of length `2`, found `3`."
        ):
            ligrec(
                adata, _CK, interactions=interactions, clusters=["foo", ("bar", "baz")]
            )
