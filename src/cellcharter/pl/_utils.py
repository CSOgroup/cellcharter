from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import Any, Mapping

import matplotlib as mpl
import numpy as np
import seaborn as sns
import squidpy as sq
from anndata import AnnData
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster import hierarchy as sch


def _heatmap(
    adata: AnnData,
    key: str,
    title: str = "",
    method: str | None = None,
    cont_cmap: str | mcolors.Colormap = "bwr",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Axes | None = None,
    n_digits: int = 2,
    **kwargs: Any,
) -> mpl.figure.Figure:

    cbar_kwargs = dict(cbar_kwargs)
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)
    else:
        fig = ax.figure

    if method is not None:
        row_order, col_order, row_link, col_link = sq.pl._utils._dendrogram(
            adata.X, method, optimal_ordering=adata.n_obs <= 1500
        )
    else:
        row_order = np.arange(len(adata.obs[key]))
        col_order = np.arange(len(adata.var_names))

    row_order = row_order[::-1]
    row_labels = adata.obs[key][row_order]
    col_labels = adata.var_names[col_order]

    data = adata[row_order, col_order].copy().X

    # row_cmap, col_cmap, row_norm, col_norm, n_cls = sq.pl._utils._get_cmap_norm(adata, key, order=(row_order, len(row_order) + col_order))
    row_cmap, col_cmap, row_norm, col_norm, n_cls = sq.pl._utils._get_cmap_norm(
        adata, key, order=(row_order, col_order)
    )
    col_norm = mcolors.BoundaryNorm(np.arange(len(col_order) + 1), col_cmap.N)

    row_sm = mpl.cm.ScalarMappable(cmap=row_cmap, norm=row_norm)
    col_sm = mpl.cm.ScalarMappable(cmap=col_cmap, norm=col_norm)

    vmin = kwargs.pop("vmin", np.nanmin(data))
    vmax = kwargs.pop("vmax", np.nanmax(data))
    vcenter = kwargs.pop("vcenter", 0)
    norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    cont_cmap = copy(plt.get_cmap(cont_cmap))
    cont_cmap.set_bad(color="grey")

    ax = sns.heatmap(
        data[::-1],
        cmap=cont_cmap,
        norm=norm,
        ax=ax,
        square=True,
        annot=np.round(data[::-1], n_digits) if annotate else False,
        cbar=False,
        **kwargs,
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    row_cats = divider.append_axes("left", size="3%", pad=0)
    col_cats = divider.append_axes("top", size="3%", pad=0)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    if method is not None:  # cluster rows but don't plot dendrogram
        col_ax = divider.append_axes("top", size="5%")
        sch.dendrogram(col_link, no_labels=True, ax=col_ax, color_threshold=0, above_threshold_color="black")
        col_ax.axis("off")

    c = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cont_cmap),
        cax=cax,
        ticks=np.linspace(norm.vmin, norm.vmax, 10),
        orientation="vertical",
        format="%0.2f",
        **cbar_kwargs,
    )

    # column labels colorbar
    c = fig.colorbar(col_sm, cax=col_cats, orientation="horizontal", ticklocation="top")
    c.set_ticks(np.arange(len(col_labels)) + 0.5)
    c.set_ticklabels(col_labels)
    (col_cats if method is None else col_ax).set_title(title)
    c.outline.set_visible(False)

    # row labels colorbar
    c = fig.colorbar(row_sm, cax=row_cats, orientation="vertical", ticklocation="left")
    c.set_ticks(np.arange(n_cls) + 0.5)
    c.set_ticklabels(row_labels)
    c.set_label(key)
    c.outline.set_visible(False)

    return fig, ax
