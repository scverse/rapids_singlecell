from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, NamedTuple

import cupy as cp
import numpy as np
import pandas as pd
from cuml.metrics import pairwise_distances
from cuml.preprocessing import MinMaxScaler
from cupy.random import choice
from cupyx.scipy.sparse import issparse as cp_issparse

# TODO(selmanozleyen): adapt which progress bar to use, probably use rapidsinglecell's progress bar (if it exists)
# from rich.progress import track
# from scipy.sparse import issparse
# from scipy.spatial.distance import cosine, mahalanobis
# from scipy.special import gammaln
# from scipy.stats import kendalltau, kstest, pearsonr, spearmanr
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import pairwise_distances, r2_score
# from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
# from sklearn.neighbors import KernelDensity
# from statsmodels.discrete.discrete_model import NegativeBinomialP

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData


class MeanVar(NamedTuple):
    mean: float
    variance: float


Metric = Literal[
    "edistance",
    # "euclidean",
    # "root_mean_squared_error",
    # "mse",
    # "mean_absolute_error",
    # "pearson_distance",
    # "spearman_distance",
    # "kendalltau_distance",
    # "cosine_distance",
    # "r2_distance",
    # "mean_pairwise",
    # "mmd",
    # "sym_kldiv",
    # "t_test",
    # "ks_test",
    # "nb_ll",
    # "classifier_proba",
    # "classifier_cp",
    # "mean_var_distribution",
    # "mahalanobis",
]


class Distance:
    """Distance class, used to compute distances between groups of cells.

    The distance metric can be specified by the user. This class also provides a
    method to compute the pairwise distances between all groups of cells.
    Currently available metrics:

    - "edistance": Energy distance (Default metric).
        In essence, it is twice the mean pairwise distance between cells of two
        groups minus the mean pairwise distance between cells within each group
        respectively. More information can be found in
        `Peidli et al. (2023) <https://doi.org/10.1101/2022.08.20.504663>`__.
    - "euclidean": euclidean distance.
        Euclidean distance between the means of cells from two groups.
    - "root_mean_squared_error": euclidean distance.
        Euclidean distance between the means of cells from two groups.
    - "mse": Pseudobulk mean squared error.
        mean squared distance between the means of cells from two groups.
    - "mean_absolute_error": Pseudobulk mean absolute distance.
        Mean absolute distance between the means of cells from two groups.
    - "pearson_distance": Pearson distance.
        Pearson distance between the means of cells from two groups.
    - "spearman_distance": Spearman distance.
        Spearman distance between the means of cells from two groups.
    - "kendalltau_distance": Kendall tau distance.
        Kendall tau distance between the means of cells from two groups.
    - "cosine_distance": Cosine distance.
        Cosine distance between the means of cells from two groups.
    - "r2_distance": coefficient of determination distance.
        Coefficient of determination distance between the means of cells from two groups.
    - "mean_pairwise": Mean pairwise distance.
        Mean of the pairwise euclidean distances between cells of two groups.
    - "mmd": Maximum mean discrepancy
        Maximum mean discrepancy between the cells of two groups.
        Here, uses linear, rbf, and quadratic polynomial MMD. For theory on MMD in single-cell applications, see
        `Lotfollahi et al. (2019) <https://doi.org/10.48550/arXiv.1910.01791>`__.
    - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        Wasserstein distance between the cells of two groups. Uses an
        OTT-JAX implementation of the Sinkhorn algorithm to compute the distance.
        For more information on the optimal transport solver, see
        `Cuturi et al. (2013) <https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf>`__.
    - "sym_kldiv": symmetrized Kullback–Leibler divergence distance.
        Kullback–Leibler divergence of the gaussian distributions between cells of two groups.
        Here we fit a gaussian distribution over one group of cells and then calculate the KL divergence on the other, and vice versa.
    - "t_test": t-test statistic.
        T-test statistic measure between cells of two groups.
    - "ks_test": Kolmogorov-Smirnov test statistic.
        Kolmogorov-Smirnov test statistic measure between cells of two groups.
    - "nb_ll": log-likelihood over negative binomial
        Average of log-likelihoods of samples of the secondary group after fitting a negative binomial distribution
        over the samples of the first group.
    - "classifier_proba": probability of a binary classifier
        Average of the classification probability of the perturbation for a binary classifier.
    - "classifier_cp": classifier class projection
        Average of the class
    - "mean_var_distribution": Distance between mean-variance distributions between cells of 2 groups.
       Mean square distance between the mean-variance distributions of cells from 2 groups using Kernel Density Estimation (KDE).
    - "mahalanobis": Mahalanobis distance between the means of cells from two groups.
        It is originally used to measure distance between a point and a distribution.
        in this context, it quantifies the difference between the mean profiles of a target group and a reference group.

    Attributes:
        metric: Name of distance metric.
        layer_key: Name of the counts to use in adata.layers.
        obsm_key: Name of embedding in adata.obsm to use.
        cell_wise_metric: Metric from scipy.spatial.distance to use for pairwise distances between single cells.

    Examples:
        >>> import pertpy as pt
        >>> adata = pt.dt.distance_example()
        >>> Distance = pt.tools.Distance(metric="edistance")
        >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
        >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
        >>> D = Distance(X, Y)
    """

    def __init__(
        self,
        metric: Metric = "edistance",
        agg_fct: Callable = cp.mean,
        layer_key: str = None,
        obsm_key: str = None,
        cell_wise_metric: str = "euclidean",
    ):
        """Initialize Distance class.

        Args:
            metric: Distance metric to use.
            agg_fct: Aggregation function to generate pseudobulk vectors.
            layer_key: Name of the counts layer containing raw counts to calculate distances for.
                              Mutually exclusive with 'obsm_key'.
                              Is not used if `None`.
            obsm_key: Name of embedding in adata.obsm to use.
                      Mutually exclusive with 'layer_key'.
                      Defaults to None, but is set to "X_pca" if not explicitly set internally.
            cell_wise_metric: Metric from scipy.spatial.distance to use for pairwise distances between single cells.
        """
        metric_fct: AbstractDistance = None
        self.aggregation_func = agg_fct
        # elif metric in ("euclidean", "root_mean_squared_error"):
        #     metric_fct = EuclideanDistance(self.aggregation_func)
        # elif metric == "mse":
        #     metric_fct = MeanSquaredDistance(self.aggregation_func)
        # elif metric == "mean_absolute_error":
        #     metric_fct = MeanAbsoluteDistance(self.aggregation_func)
        # elif metric == "pearson_distance":
        #     metric_fct = PearsonDistance(self.aggregation_func)
        # elif metric == "spearman_distance":
        #     metric_fct = SpearmanDistance(self.aggregation_func)
        # elif metric == "kendalltau_distance":
        #     metric_fct = KendallTauDistance(self.aggregation_func)
        # elif metric == "cosine_distance":
        #     metric_fct = CosineDistance(self.aggregation_func)
        # elif metric == "r2_distance":
        #     metric_fct = R2ScoreDistance(self.aggregation_func)
        # elif metric == "mean_pairwise":
        #     metric_fct = MeanPairwiseDistance()
        # elif metric == "mmd":
        #     metric_fct = MMD()
        # elif metric == "sym_kldiv":
        #     metric_fct = SymmetricKLDivergence()
        # elif metric == "t_test":
        #     metric_fct = TTestDistance()
        # elif metric == "ks_test":
        #     metric_fct = KSTestDistance()
        # elif metric == "nb_ll":
        #     metric_fct = NBLL()
        # elif metric == "classifier_proba":
        #     metric_fct = ClassifierProbaDistance()
        # elif metric == "classifier_cp":
        #     metric_fct = ClassifierClassProjection()
        # elif metric == "mean_var_distribution":
        #     metric_fct = MeanVarDistributionDistance()
        # elif metric == "mahalanobis":
        #     metric_fct = MahalanobisDistance(self.aggregation_func)
        if metric == "edistance":
            metric_fct = Edistance()
        else:
            raise ValueError(f"Metric {metric} not recognized.")
        self.metric_fct = metric_fct

        if layer_key and obsm_key:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time.\nPlease provide only one of the two keys."
            )
        if not layer_key and not obsm_key:
            obsm_key = "X_pca"
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self.metric = metric
        self.cell_wise_metric = cell_wise_metric

    def __call__(
        self,
        X: cp.ndarray | cp.sparse.spmatrix,
        Y: cp.ndarray | cp.sparse.spmatrix,
        **kwargs,
    ) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            kwargs: Passed to the metric function.

        Returns:
            float: Distance between X and Y.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
            >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
            >>> D = Distance(X, Y)
        """
        if cp_issparse(X):
            X = X.toarray()
        if cp_issparse(Y):
            Y = Y.toarray()

        if len(X) == 0 or len(Y) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        return self.metric_fct(X, Y, **kwargs)

    def bootstrap(
        self,
        X: cp.ndarray,
        Y: cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
        **kwargs,
    ) -> MeanVar:
        """Bootstrap computation of mean and variance of the distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            n_bootstrap: Number of bootstrap samples.
            random_state: Random state for bootstrapping.
            **kwargs: Passed to the metric function.

        Returns:
            Mean and variance of distance between X and Y.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
            >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
            >>> D = Distance.bootstrap(X, Y)
        """
        return self._bootstrap_mode(
            X,
            Y,
            n_bootstraps=n_bootstrap,
            random_state=random_state,
            **kwargs,
        )

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        groups: list[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Get pairwise distances between groups of cells.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            groups: List of groups to compute pairwise distances for.
                    If None, uses all groups.
            bootstrap: Whether to bootstrap the distance.
            n_bootstrap: Number of bootstrap samples.
            random_state: Random state for bootstrapping.
            show_progressbar: Whether to show progress bar.
            n_jobs: Number of cores to use. Defaults to -1 (all).
            kwargs: Additional keyword arguments passed to the metric function.

        Returns:
            :class:`pandas.DataFrame`: Dataframe with pairwise distances.
            tuple[:class:`pandas.DataFrame`, :class:`pandas.DataFrame`]: Two Dataframes, one for the mean and one for the variance of pairwise distances.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> pairwise_df = Distance.pairwise(adata, groupby="perturbation")
        """
        groups = adata.obs[groupby].unique() if groups is None else groups
        grouping = adata.obs[groupby].copy()
        df = pd.DataFrame(index=groups, columns=groups, dtype=float)
        if bootstrap:
            df_var = pd.DataFrame(index=groups, columns=groups, dtype=float)
        # fct = track if show_progressbar else lambda iterable: iterable
        fct = lambda iterable: iterable  # see TODO above about progress bar

        # Some metrics are able to handle precomputed distances. This means that
        # the pairwise distances between all cells are computed once and then
        # passed to the metric function. This is much faster than computing the
        # pairwise distances for each group separately. Other metrics are not
        # able to handle precomputed distances such as the PseudobulkDistance.
        if self.metric_fct.accepts_precomputed:
            # Precompute the pairwise distances if needed
            if (
                f"{self.obsm_key}_{self.cell_wise_metric}_predistances"
                not in adata.obsp
            ):
                self.precompute_distances(adata, n_jobs=n_jobs, **kwargs)
            pwd = adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"]
            for index_x, group_x in enumerate(fct(groups)):
                idx_x = grouping == group_x
                for group_y in groups[index_x:]:  # type: ignore
                    # subset the pairwise distance matrix to the two groups
                    idx_y = grouping == group_y
                    sub_pwd = pwd[idx_x | idx_y, :][:, idx_x | idx_y]
                    sub_idx = grouping[idx_x | idx_y] == group_x
                    if not bootstrap:
                        if group_x == group_y:
                            dist = 0.0
                        else:
                            dist = self.metric_fct.from_precomputed(
                                sub_pwd, sub_idx, **kwargs
                            )
                        df.loc[group_x, group_y] = dist
                        df.loc[group_y, group_x] = dist

                    else:
                        bootstrap_output = self._bootstrap_mode_precomputed(
                            sub_pwd,
                            sub_idx,
                            n_bootstraps=n_bootstrap,
                            random_state=random_state,
                            **kwargs,
                        )
                        # In the bootstrap case, distance of group to itself is a mean and can be non-zero
                        df.loc[group_x, group_y] = df.loc[group_y, group_x] = (
                            bootstrap_output.mean
                        )
                        df_var.loc[group_x, group_y] = df_var.loc[group_y, group_x] = (
                            bootstrap_output.variance
                        )
        else:
            embedding = (
                adata.layers[self.layer_key]
                if self.layer_key
                else adata.obsm[self.obsm_key].copy()
            )
            for index_x, group_x in enumerate(fct(groups)):
                cells_x = embedding[cp.asarray(grouping == group_x)].copy()
                for group_y in groups[index_x:]:  # type: ignore
                    cells_y = embedding[cp.asarray(grouping == group_y)].copy()
                    if not bootstrap:
                        # By distance axiom, the distance between a group and itself is 0
                        dist = (
                            0.0
                            if group_x == group_y
                            else self(cells_x, cells_y, **kwargs)
                        )

                        df.loc[group_x, group_y] = dist
                        df.loc[group_y, group_x] = dist
                    else:
                        bootstrap_output = self.bootstrap(
                            cells_x,
                            cells_y,
                            n_bootstrap=n_bootstrap,
                            random_state=random_state,
                            **kwargs,
                        )
                        # In the bootstrap case, distance of group to itself is a mean and can be non-zero
                        df.loc[group_x, group_y] = df.loc[group_y, group_x] = (
                            bootstrap_output.mean
                        )
                        df_var.loc[group_x, group_y] = df_var.loc[group_y, group_x] = (
                            bootstrap_output.variance
                        )

        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric}"

        if not bootstrap:
            return df
        else:
            df = df.fillna(0)
            df_var.index.name = groupby
            df_var.columns.name = groupby
            df_var = df_var.fillna(0)
            df_var.name = f"pairwise {self.metric} variance"

            return df, df_var

    def precompute_distances(self, adata: AnnData, n_jobs: int = -1) -> None:
        """Precompute pairwise distances between all cells, writes to adata.obsp.

        The precomputed distances are stored in adata.obsp under the key
        '{self.obsm_key}_{cell_wise_metric}_predistances', as they depend on
        both the cell-wise metric and the embedding used.

        Args:
            adata: Annotated data matrix.
            n_jobs: Number of cores to use. Defaults to -1 (all).

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> distance = pt.tools.Distance(metric="edistance")
            >>> distance.precompute_distances(adata)
        """
        cells = (
            adata.layers[self.layer_key]
            if self.layer_key
            else adata.obsm[self.obsm_key].copy()
        )
        pwd = pairwise_distances(
            cells, cells, metric=self.cell_wise_metric, n_jobs=n_jobs
        )
        adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"] = pwd

    def compare_distance(
        self,
        pert: cp.ndarray,
        pred: cp.ndarray,
        ctrl: cp.ndarray,
        mode: Literal["simple", "scaled"] = "simple",
        fit_to_pert_and_ctrl: bool = False,
        **kwargs,
    ) -> float:
        """Compute the score of simulating a perturbation.

        Args:
            pert: Real perturbed data.
            pred: Simulated perturbed data.
            ctrl: Control data
            mode: Mode to use.
            fit_to_pert_and_ctrl: Scales data based on both `pert` and `ctrl` if True, otherwise only on `ctrl`.
            kwargs: Additional keyword arguments passed to the metric function.
        """
        if mode == "simple":
            pass  # nothing to be done
        elif mode == "scaled":
            scaler = MinMaxScaler().fit(
                np.vstack((pert, ctrl)) if fit_to_pert_and_ctrl else ctrl
            )
            pred = scaler.transform(pred)
            pert = scaler.transform(pert)
        else:
            raise ValueError(f"Unknown mode {mode}. Please choose simple or scaled.")

        d1 = self.metric_fct(pert, pred, **kwargs)
        d2 = self.metric_fct(ctrl, pred, **kwargs)
        return d1 / d2

    def _bootstrap_mode(
        self, X, Y, n_bootstraps=100, random_state=0, **kwargs
    ) -> MeanVar:
        # rng = np.random.default_rng(random_state)

        distances = []
        for _ in range(n_bootstraps):
            X_bootstrapped = X[choice(a=X.shape[0], size=X.shape[0], replace=True)]
            Y_bootstrapped = Y[choice(a=Y.shape[0], size=X.shape[0], replace=True)]

            distance = self(X_bootstrapped, Y_bootstrapped, **kwargs)
            distances.append(distance.get())

        mean = np.mean(distances)
        variance = np.var(distances)
        return MeanVar(mean=mean, variance=variance)

    def _bootstrap_mode_precomputed(
        self, sub_pwd, sub_idx, n_bootstraps=100, random_state=0, **kwargs
    ) -> MeanVar:
        rng = np.random.default_rng(random_state)

        distances = []
        for _ in range(n_bootstraps):
            # To maintain the number of cells for both groups (whatever balancing they may have),
            # we sample the positive and negative indices separately
            bootstrap_pos_idx = rng.choice(
                a=sub_idx[sub_idx].index, size=sub_idx[sub_idx].size, replace=True
            )
            bootstrap_neg_idx = rng.choice(
                a=sub_idx[~sub_idx].index, size=sub_idx[~sub_idx].size, replace=True
            )
            bootstrap_idx = np.concatenate([bootstrap_pos_idx, bootstrap_neg_idx])
            bootstrap_idx_nrs = sub_idx.index.get_indexer(bootstrap_idx)

            bootstrap_sub_idx = sub_idx[bootstrap_idx]
            bootstrap_sub_pwd = sub_pwd[bootstrap_idx_nrs, :][:, bootstrap_idx_nrs]

            distance = self.metric_fct.from_precomputed(
                bootstrap_sub_pwd, bootstrap_sub_idx, **kwargs
            )
            distances.append(distance)

        mean = np.mean(distances)
        variance = np.var(distances)
        return MeanVar(mean=mean, variance=variance)

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str | None = None,
        groups: list[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Get distances between one selected cell group and the remaining other cell groups.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            selected_group: Group to compute pairwise distances to all other.
            groups: List of groups to compute distances to selected_group for.
                    If None, uses all groups.
            bootstrap: Whether to bootstrap the distance.
            n_bootstrap: Number of bootstrap samples.
            random_state: Random state for bootstrapping.
            show_progressbar: Whether to show progress bar.
            n_jobs: Number of cores to use. Defaults to -1 (all).
            kwargs: Additional keyword arguments passed to the metric function.

        Returns:
            :class:`pandas.DataFrame`: Dataframe with distances of groups to selected_group.
            tuple[:class:`pandas.DataFrame`, :class:`pandas.DataFrame`]: Two Dataframes, one for the mean and one for the variance of distances of groups to selected_group.


        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> pairwise_df = Distance.onesided_distances(adata, groupby="perturbation", selected_group="control")
        """
        if self.metric == "classifier_cp":
            if bootstrap:
                raise NotImplementedError(
                    "Currently, ClassifierClassProjection does not support bootstrapping."
                )
            return self.metric_fct.onesided_distances(  # type: ignore
                adata,
                groupby,
                selected_group,
                groups,
                show_progressbar,
                n_jobs,
                **kwargs,
            )

        groups = adata.obs[groupby].unique() if groups is None else groups
        grouping = adata.obs[groupby].copy()
        df = pd.Series(index=groups, dtype=float)
        if bootstrap:
            df_var = pd.Series(index=groups, dtype=float)
        # fct = track if show_progressbar else lambda iterable: iterable
        fct = (
            lambda iterable: iterable
        )  # see TODO at the top of the file about progress bar

        # Some metrics are able to handle precomputed distances. This means that
        # the pairwise distances between all cells are computed once and then
        # passed to the metric function. This is much faster than computing the
        # pairwise distances for each group separately. Other metrics are not
        # able to handle precomputed distances such as the PseudobulkDistance.
        if self.metric_fct.accepts_precomputed:
            # Precompute the pairwise distances if needed
            if (
                f"{self.obsm_key}_{self.cell_wise_metric}_predistances"
                not in adata.obsp
            ):
                self.precompute_distances(adata, n_jobs=n_jobs, **kwargs)
            pwd = adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"]
            for group_x in fct(groups):
                idx_x = grouping == group_x
                group_y = selected_group
                if group_x == group_y:
                    df.loc[group_x] = 0.0  # by distance axiom
                else:
                    idx_y = grouping == group_y
                    # subset the pairwise distance matrix to the two groups
                    sub_pwd = pwd[idx_x | idx_y, :][:, idx_x | idx_y]
                    sub_idx = grouping[idx_x | idx_y] == group_x
                    if not bootstrap:
                        dist = self.metric_fct.from_precomputed(
                            sub_pwd, sub_idx, **kwargs
                        )
                        df.loc[group_x] = dist
                    else:
                        bootstrap_output = self._bootstrap_mode_precomputed(
                            sub_pwd,
                            sub_idx,
                            n_bootstraps=n_bootstrap,
                            random_state=random_state,
                            **kwargs,
                        )
                        df.loc[group_x] = bootstrap_output.mean
                        df_var.loc[group_x] = bootstrap_output.variance
        else:
            embedding = (
                adata.layers[self.layer_key]
                if self.layer_key
                else adata.obsm[self.obsm_key].copy()
            )
            for group_x in fct(groups):
                cells_x = embedding[np.asarray(grouping == group_x)].copy()
                group_y = selected_group
                cells_y = embedding[np.asarray(grouping == group_y)].copy()
                if not bootstrap:
                    # By distance axiom, the distance between a group and itself is 0
                    dist = (
                        0.0 if group_x == group_y else self(cells_x, cells_y, **kwargs)
                    )
                    df.loc[group_x] = dist
                else:
                    bootstrap_output = self.bootstrap(
                        cells_x,
                        cells_y,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                        **kwargs,
                    )
                    # In the bootstrap case, distance of group to itself is a mean and can be non-zero
                    df.loc[group_x] = bootstrap_output.mean
                    df_var.loc[group_x] = bootstrap_output.variance
        df.index.name = groupby
        df.name = f"{self.metric} to {selected_group}"
        if not bootstrap:
            return df
        else:
            df_var.index.name = groupby
            df_var = df_var.fillna(0)
            df_var.name = f"pairwise {self.metric} variance to {selected_group}"

            return df, df_var


class AbstractDistance(ABC):
    """Abstract class of distance metrics between two sets of vectors."""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed: bool = None

    @abstractmethod
    def __call__(self, X: cp.ndarray, Y: cp.ndarray, **kwargs) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            kwargs: Passed to the metrics function.

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")

    @abstractmethod
    def from_precomputed(self, P: cp.ndarray, idx: cp.ndarray, **kwargs) -> float:
        """Compute a distance between vectors X and Y with precomputed distances.

        Args:
            P: Pairwise distance matrix of shape (n_samples, n_samples).
            idx: Boolean array of shape (n_samples,) indicating which samples belong to X (or Y, since each metric is symmetric).
            kwargs: Passed to the metrics function.

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")


class Edistance(AbstractDistance):
    """Edistance metric."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True
        self.cell_wise_metric = "euclidean"

    def __call__(self, X: cp.ndarray, Y: cp.ndarray, **kwargs) -> float:
        sigma_X = pairwise_distances(
            X, X, metric=self.cell_wise_metric, **kwargs
        ).mean()
        sigma_Y = pairwise_distances(
            Y, Y, metric=self.cell_wise_metric, **kwargs
        ).mean()
        delta = pairwise_distances(X, Y, metric=self.cell_wise_metric, **kwargs).mean()
        return 2 * delta - sigma_X - sigma_Y

    def from_precomputed(self, P: cp.ndarray, idx: cp.ndarray, **kwargs) -> float:
        sigma_X = P[idx, :][:, idx].mean()
        sigma_Y = P[~idx, :][:, ~idx].mean()
        delta = P[idx, :][:, ~idx].mean()
        return 2 * delta - sigma_X - sigma_Y
