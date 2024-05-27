from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import Any, Mapping

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import squidpy as sq
from anndata import AnnData
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scanpy.pl._dotplot import DotPlot
from scipy.cluster import hierarchy as sch
from squidpy._constants._pkg_constants import Key
from squidpy.pl._color_utils import Palette_t

try:
    from matplotlib import colormaps as cm
except ImportError:
    from matplotlib import cm


def _get_cmap_norm(
    adata: AnnData,
    key: str,
    order: tuple[list[int], list[int]] | None | None = None,
) -> tuple[mcolors.ListedColormap, mcolors.ListedColormap, mcolors.BoundaryNorm, mcolors.BoundaryNorm, int]:
    n_rows = n_cols = adata.obs[key].nunique()

    colors = adata.uns[Key.uns.colors(key)]

    if order is not None:
        row_order, col_order = order
        row_colors = [colors[i] for i in row_order]
        col_colors = [colors[i] for i in col_order]

        n_rows = len(row_order)
        n_cols = len(col_order)
    else:
        row_colors = col_colors = colors

    row_cmap = mcolors.ListedColormap(row_colors)
    col_cmap = mcolors.ListedColormap(col_colors)
    row_norm = mcolors.BoundaryNorm(np.arange(n_rows + 1), row_cmap.N)
    col_norm = mcolors.BoundaryNorm(np.arange(n_cols + 1), col_cmap.N)

    return row_cmap, col_cmap, row_norm, col_norm, n_rows


def _heatmap(
    adata: AnnData,
    key: str,
    rows: list[str] | None = None,
    cols: list[str] | None = None,
    title: str = "",
    method: str | None = None,
    cont_cmap: str | mcolors.Colormap = "bwr",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Axes | None = None,
    n_digits: int = 2,
    show_cols: bool = True,
    show_rows: bool = True,
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
        row_order = (
            np.arange(len(adata.obs[key]))
            if rows is None
            else np.argwhere(adata.obs.index.isin(np.array(rows).astype(str))).flatten()
        )
        col_order = (
            np.arange(len(adata.var_names))
            if cols is None
            else np.argwhere(adata.var_names.isin(np.array(cols).astype(str))).flatten()
        )

    row_order = row_order[::-1]
    row_labels = adata.obs[key][row_order]
    col_labels = adata.var_names[col_order]

    data = adata[row_order, col_order].copy().X

    # row_cmap, col_cmap, row_norm, col_norm, n_cls = sq.pl._utils._get_cmap_norm(adata, key, order=(row_order, len(row_order) + col_order))
    row_cmap, col_cmap, row_norm, col_norm, n_cls = _get_cmap_norm(adata, key, order=(row_order, col_order))
    col_norm = mcolors.BoundaryNorm(np.arange(len(col_order) + 1), col_cmap.N)

    row_sm = mpl.cm.ScalarMappable(cmap=row_cmap, norm=row_norm)
    col_sm = mpl.cm.ScalarMappable(cmap=col_cmap, norm=col_norm)

    vmin = kwargs.pop("vmin", np.nanmin(data))
    vmax = kwargs.pop("vmax", np.nanmax(data))
    vcenter = kwargs.pop("vcenter", 0)
    norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    cont_cmap = copy(cm.get_cmap(cont_cmap))
    cont_cmap.set_bad(color="grey")

    annot = np.round(data[::-1], n_digits).astype(str) if annotate else None
    if "significant" in adata.layers:
        significant = adata.layers["significant"].astype(str)
        annot = np.char.add(np.empty_like(data[::-1], dtype=str), significant[row_order[:, None], col_order][::-1])

    ax = sns.heatmap(
        data[::-1],
        cmap=cont_cmap,
        norm=norm,
        ax=ax,
        square=True,
        annot=annot,
        cbar=False,
        fmt="",
        **kwargs,
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    row_cats = divider.append_axes("left", size=0.1, pad=0.1)
    col_cats = divider.append_axes("bottom", size=0.1, pad=0.1)
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
    c = fig.colorbar(col_sm, cax=col_cats, orientation="horizontal", ticklocation="bottom")

    if rows == cols or show_cols is False:
        c.set_ticks([])
        c.set_ticklabels([])
    else:
        c.set_ticks(np.arange(len(col_labels)) + 0.5)
        c.set_ticklabels(col_labels)
        if np.any([len(l) > 3 for l in col_labels]):
            c.ax.tick_params(rotation=90)
    c.outline.set_visible(False)

    # row labels colorbar
    c = fig.colorbar(row_sm, cax=row_cats, orientation="vertical", ticklocation="left")
    if show_rows is False:
        c.set_ticks([])
        c.set_ticklabels([])
    else:
        c.set_ticks(np.arange(n_cls) + 0.5)
        c.set_ticklabels(row_labels)
        c.set_label(key)
    c.outline.set_visible(False)

    ax.set_title(title)

    return fig, ax


def _reorder(values, order, axis=1):
    if axis == 0:
        values = values.iloc[order, :]
    elif axis == 1:
        values = values.iloc[:, order]
    else:
        raise ValueError("The axis parameter accepts only values 0 and 1.")
    return values


def _clip(values, min_threshold=None, max_threshold=None, new_min=None, new_max=None, new_middle=None):
    values_clipped = values.copy()
    if new_middle is not None:
        values_clipped[:] = new_middle
    if min_threshold is not None:
        values_clipped[values < min_threshold] = new_min if new_min is not None else min_threshold
    if max_threshold is not None:
        values_clipped[values > max_threshold] = new_max if new_max is not None else max_threshold
    return values_clipped


def _dotplot(
    adata: AnnData,
    x_key: str,
    y_key: str,
    values: np.ndarray,
    abs_values: bool = False,
    size_threshold: tuple[float, float] | tuple[None, None] = (None, None),
    color_threshold: tuple[float, float] = (-1, 1),
    figsize: tuple[float, float] | None = None,
    cmap: str | Palette_t = "bwr",
    size_title: str = "log2 FC",
    dot_scale: float = 1,
    cluster_y: bool = True,
    **kwargs,
):
    values_color = _clip(
        values, min_threshold=color_threshold[0], max_threshold=color_threshold[1], new_min=-1, new_max=1
    )
    values_color[(values < 0) & (values > color_threshold[0])] = 0  # -0.3
    values_color[(values > 0) & (values < color_threshold[1])] = 0  # 0.3

    if cluster_y is True:
        order = sp.cluster.hierarchy.dendrogram(
            sp.cluster.hierarchy.linkage(values_color.T, method="complete"), no_plot=True
        )["leaves"]
        values = _reorder(values, order, axis=1)
        values_color = _reorder(values_color, order, axis=1)

    one_hot_encoded = pd.get_dummies(adata.obs[y_key])

    adata_obs = AnnData(one_hot_encoded, dtype=np.uint8, obs=adata.obs)

    values_size = _clip(values, size_threshold[0], size_threshold[1])
    values_size = pd.DataFrame(
        (mcolors.TwoSlopeNorm(vcenter=0, vmin=size_threshold[0], vmax=size_threshold[1])(values_size) - 0.5) * 2,
        columns=values_size.columns,
        index=values_size.index,
    )

    if abs_values:
        print("Warning: label for depletion/enrichment to be implemented.")
        values_size = np.abs(values_size)

    if figsize is None:
        figsize = (10, 10 * values.shape[1] / values.shape[0])

    dp = MyDotPlot(
        adata_obs,
        adata_obs.var_names,
        groupby=x_key,
        dot_color_df=values_color,
        dot_size_df=values_size,
        figsize=figsize,
        **kwargs,
    )
    dp.max_value = np.max(values_size.values)
    dp.color_threshold = color_threshold[1]
    dp.size_threshold = size_threshold[1]

    dp.swap_axes()
    dp = dp.style(
        cmap=cmap,
        largest_dot=dp.largest_dot * dot_scale,
        dot_edge_lw=DotPlot.DEFAULT_DOT_EDGELW,
    )
    dp = dp.legend(show_colorbar=False, size_title=size_title)
    return dp


class MyDotPlot(DotPlot):
    """Modified version :class:`scanpy.pl.DotPlot`."""

    def _plot_size_legend(self, size_legend_ax: Axes):
        size_range = np.linspace(self.dot_min, self.size_threshold, 3)
        if self.dot_min == 0:
            size_range[0] += self.size_threshold / 10

        size = (size_range / (self.size_threshold - self.dot_min)) ** self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot
        # plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            c=["gray" if s < self.color_threshold else (0.705673158, 0.01555616, 0.150232812, 1.0) for s in size_range],
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{np.round((x * self.max_value), decimals=2)}" for x in size_range]
        labels[-1] = f">{labels[-1]}"
        size_legend_ax.set_xticklabels(labels, fontsize="x-small")

        # remove y ticks and labels
        size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

        # remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size="small")

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    # ToDo: need to find a way to get get_axes()['mainplot_ax'] without showing the plot
    def _rotate_xlabels(self, ax):
        ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center", minor=False)


def adjust_box_widths(g, fac):
    """Adjust the widths of a seaborn-generated boxplot."""
    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
