from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import rcParams
from matplotlib.axes import Axes
from squidpy._docs import d
from squidpy.gr._utils import _assert_categorical_obs
from squidpy.pl._color_utils import Palette_t, _maybe_set_colors
from squidpy.pl._graph import _get_data
from squidpy.pl._spatial_utils import _panel_grid

from cellcharter.pl._utils import _heatmap


def _plot_nhood_enrichment(
    adata: AnnData,
    nhood_enrichment_values: dict,
    cluster_key: str,
    row_groups: str | None = None,
    col_groups: str | None = None,
    annotate: bool = False,
    n_digits: int = 2,
    significance: float | None = None,
    method: str | None = None,
    title: str | None = "Neighborhood enrichment",
    cmap: str = "bwr",
    palette: Palette_t = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
):
    enrichment = nhood_enrichment_values["enrichment"]
    adata_enrichment = AnnData(X=enrichment.astype(np.float32))
    adata_enrichment.obs[cluster_key] = pd.Categorical(enrichment.index)

    if significance is not None:
        if "pvalue" not in nhood_enrichment_values:
            warnings.warn(
                "Significance requires gr.nhood_enrichment to be run with pvalues=True. Ignoring significance.",
                UserWarning,
                stacklevel=2,
            )
        else:
            adata_enrichment.layers["significant"] = np.empty_like(enrichment, dtype=str)
            adata_enrichment.layers["significant"][nhood_enrichment_values["pvalue"].values <= significance] = "*"

    _maybe_set_colors(source=adata, target=adata_enrichment, key=cluster_key, palette=palette)

    if figsize is None:
        figsize = list(adata_enrichment.shape[::-1])

        if row_groups is not None:
            figsize[1] = len(row_groups)

        if col_groups is not None:
            figsize[0] = len(col_groups)

        figsize = tuple(figsize)

    _heatmap(
        adata_enrichment,
        key=cluster_key,
        rows=row_groups,
        cols=col_groups,
        annotate=annotate,
        n_digits=n_digits,
        method=method,
        title=title,
        cont_cmap=cmap,
        figsize=figsize,
        dpi=dpi,
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        **kwargs,
    )


@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    row_groups: list[str] | None = None,
    col_groups: list[str] | None = None,
    min_freq: float | None = None,
    annotate: bool = False,
    transpose: bool = False,
    method: str | None = None,
    title: str | None = "Neighborhood enrichment",
    cmap: str = "bwr",
    palette: Palette_t = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    ax: Axes | None = None,
    n_digits: int = 2,
    significance: float | None = None,
    save: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    A modified version of squidpy's function for `plotting neighborhood enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.pl.nhood_enrichment.html>`_.

    The enrichment is computed by :func:`cellcharter.gr.nhood_enrichment`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    row_groups
        Restrict the rows to these groups. If `None`, all groups are plotted.
    col_groups
        Restrict the columns to these groups. If `None`, all groups are plotted.
    %(heatmap_plotting)s

    n_digits
        The number of digits of the number in the annotations.
    significance
        Mark the values that are below this threshold with a star. If `None`, no significance is computed. It requires ``gr.nhood_enrichment`` to be run with ``pvalues=True``.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.text`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    nhood_enrichment_values = _get_data(adata, cluster_key=cluster_key, func_name="nhood_enrichment").copy()
    nhood_enrichment_values["enrichment"][np.isinf(nhood_enrichment_values["enrichment"])] = np.nan

    if transpose:
        nhood_enrichment_values["enrichment"] = nhood_enrichment_values["enrichment"].T

    if min_freq is not None:
        frequency = adata.obs[cluster_key].value_counts(normalize=True)
        nhood_enrichment_values["enrichment"].loc[frequency[frequency < min_freq].index] = np.nan
        nhood_enrichment_values["enrichment"].loc[:, frequency[frequency < min_freq].index] = np.nan

    _plot_nhood_enrichment(
        adata,
        nhood_enrichment_values,
        cluster_key,
        row_groups=row_groups,
        col_groups=col_groups,
        annotate=annotate,
        method=method,
        title=title,
        cmap=cmap,
        palette=palette,
        cbar_kwargs=cbar_kwargs,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        n_digits=n_digits,
        significance=significance,
        **kwargs,
    )

    if save is not None:
        plt.savefig(save, bbox_inches="tight")


@d.dedent
def diff_nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    condition_key: str,
    condition_groups: list[str] | None = None,
    hspace: float = 0.25,
    wspace: float | None = None,
    ncols: int = 1,
    **nhood_kwargs: Any,
) -> None:
    r"""
    Plot the difference in neighborhood enrichment between conditions.

    The difference is computed by :func:`cellcharter.gr.diff_nhood_enrichment`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    condition_key
        Key in ``adata.obs`` that stores the sample condition (e.g., normal vs disease).
    condition_groups
        Restrict the conditions to these clusters. If `None`, all groups are plotted.
    hspace
        Height space between panels.
    wspace
        Width space between panels.
    ncols
        Number of panels per row.
    nhood_kwargs
        Keyword arguments for :func:`cellcharter.pl.nhood_enrichment`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_categorical_obs(adata, key=condition_key)

    conditions = adata.obs[condition_key].cat.categories if condition_groups is None else condition_groups

    if nhood_kwargs is None:
        nhood_kwargs = {}

    cmap = nhood_kwargs.pop("cmap", "PRGn_r")
    save = nhood_kwargs.pop("save", None)

    n_combinations = len(conditions) * (len(conditions) - 1) // 2

    figsize = nhood_kwargs.get("figsize", rcParams["figure.figsize"])

    # Plot neighborhood enrichment for each condition pair as a subplot
    _, grid = _panel_grid(
        num_panels=n_combinations,
        hspace=hspace,
        wspace=0.75 / figsize[0] + 0.02 if wspace is None else wspace,
        ncols=ncols,
        dpi=nhood_kwargs.get("dpi", rcParams["figure.dpi"]),
        figsize=nhood_kwargs.get("dpi", figsize),
    )

    axs = [plt.subplot(grid[c]) for c in range(n_combinations)]

    for i, (condition1, condition2) in enumerate(combinations(conditions, 2)):
        if f"{condition1}_{condition2}" not in adata.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"]:
            nhood_enrichment_values = dict(
                adata.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"][f"{condition2}_{condition1}"]
            )
            nhood_enrichment_values["enrichment"] = -nhood_enrichment_values["enrichment"]
        else:
            nhood_enrichment_values = adata.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"][
                f"{condition1}_{condition2}"
            ]

        _plot_nhood_enrichment(
            adata,
            nhood_enrichment_values,
            cluster_key,
            cmap=cmap,
            ax=axs[i],
            title=f"{condition1} vs {condition2}",
            show_cols=i >= n_combinations - ncols,  # Show column labels only the last subplot of each grid column
            show_rows=i % ncols == 0,  # Show row labels only for the first subplot of each grid row
            **nhood_kwargs,
        )

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
