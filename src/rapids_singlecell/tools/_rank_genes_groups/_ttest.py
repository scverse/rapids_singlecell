from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from ._core import _RankGenes


def t_test(
    rg: _RankGenes, method: Literal["t-test", "t-test_overestim_var"]
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Compute t-test statistics using Welch's t-test."""
    from scipy import stats

    rg._basic_stats()

    for group_index, (mask_obs, mean_group, var_group) in enumerate(
        zip(rg.groups_masks_obs, rg.means, rg.vars, strict=True)
    ):
        if rg.ireference is not None and group_index == rg.ireference:
            continue

        ns_group = np.count_nonzero(mask_obs)

        if rg.ireference is not None:
            mean_rest = rg.means[rg.ireference]
            var_rest = rg.vars[rg.ireference]
            ns_other = np.count_nonzero(rg.groups_masks_obs[rg.ireference])
        else:
            mean_rest = rg.means_rest[group_index]
            var_rest = rg.vars_rest[group_index]
            ns_other = rg.X.shape[0] - ns_group

        if method == "t-test":
            ns_rest = ns_other
        elif method == "t-test_overestim_var":
            # Hack for overestimating the variance for small groups
            ns_rest = ns_group
        else:
            msg = "Method does not exist."
            raise ValueError(msg)

        # Welch's t-test using pre-computed stats
        with np.errstate(invalid="ignore"):
            scores, pvals = stats.ttest_ind_from_stats(
                mean1=mean_group,
                std1=np.sqrt(var_group),
                nobs1=ns_group,
                mean2=mean_rest,
                std2=np.sqrt(var_rest),
                nobs2=ns_rest,
                equal_var=False,  # Welch's
            )

        # Handle NaN values (when means are the same and vars are 0)
        scores[np.isnan(scores)] = 0
        pvals[np.isnan(pvals)] = 1

        yield group_index, scores, pvals
