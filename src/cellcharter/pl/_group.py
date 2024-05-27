from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from squidpy._docs import d
from squidpy.gr._utils import _assert_categorical_obs
from squidpy.pl._color_utils import Palette_t, _get_palette, _maybe_set_colors

try:
    from matplotlib.colormaps import get_cmap
except ImportError:
    from matplotlib.pyplot import get_cmap

from cellcharter.gr._group import _proportion
from cellcharter.pl._utils import _dotplot


@d.dedent
def proportion(
    adata: AnnData,
    group_key: str,
    label_key: str,
    groups: list | None = None,
    labels: list | None = None,
    rotation_xlabel: int = 45,
    ncols: int = 1,
    normalize: bool = True,
    palette: Palette_t = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    **kwargs,
) -> None:
    """
    Plot the proportion of `y_key` in `x_key`.

    Parameters
    ----------
    %(adata)s
    group_key
        Key in :attr:`anndata.AnnData.obs` where groups are stored.
    label_key
        Key in :attr:`anndata.AnnData.obs` where labels are stored.
    groups
        List of groups to plot.
    labels
        List of labels to plot.
    rotation_xlabel
        Rotation in degrees of the ticks of the x axis.
    ncols
        Number of columns for the legend.
    normalize
        If `True` use relative frequencies, outherwise use counts.
    palette
        Categorical colormap for the clusters.
        If ``None``, use :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_colors']``, if available.
    %(plotting)s
    kwargs
        Keyword arguments for :func:`pandas.DataFrame.plot.bar`.
    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=group_key)
    _assert_categorical_obs(adata, key=label_key)
    _maybe_set_colors(source=adata, target=adata, key=label_key, palette=palette)

    clusters = adata.obs[label_key].cat.categories
    palette = _get_palette(adata, cluster_key=label_key, categories=clusters)

    df = _proportion(adata=adata, id_key=group_key, val_key=label_key, normalize=normalize)
    df = df[df.columns[::-1]]

    if groups is not None:
        df = df.loc[groups, :]

    if labels is not None:
        df = df.loc[:, labels]

    plt.figure(dpi=dpi)
    ax = df.plot.bar(stacked=True, figsize=figsize, color=palette, rot=rotation_xlabel, ax=plt.gca(), **kwargs)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[::-1], labels[::-1], loc="center left", ncol=ncols, bbox_to_anchor=(1.0, 0.5))

    if save:
        plt.savefig(save, bbox_extra_artists=(lgd, lgd), bbox_inches="tight")


@d.dedent
def enrichment(
    adata: AnnData,
    group_key: str,
    label_key: str,
    size_threshold: float | None = None,
    color_threshold: float = 1,
    legend_title: str | None = None,
    dot_scale: float = 1,
    cluster_labels: bool = True,
    groups: list | None = None,
    labels: list | None = None,
    palette: Palette_t | matplotlib.colors.ListedColormap | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
    **kwargs,
):
    """
    Plot a dotplot of the enrichment of `y_key` in `x_key`.

    This functions is based on a modified version of :func:`scanpy.pl.dotplot`.

    Parameters
    ----------
    %(adata)s
    group_key
        Key in :attr:`anndata.AnnData.obs` where groups are stored.
    label_key
        Key in :attr:`anndata.AnnData.obs` where labels are stored.
    size_threshold
        Threshold for the size of the dots. Enrichments with value above this threshold will have all the same size.
    color_threshold
        Threshold to mark enrichments as significant.
    legend_title
        Title for the size legend.
    dot_scale
        Scale of the dots.
    cluster_groups
        If `True`, display labels ordered according to hierarchical clustering.
    groups
        The groups for which to show the enrichment.
    labels
        The labels for which to show the enrichment.
    palette
        Colormap for the enrichment values.
    %(plotting)s
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.scatter`.
    """
    if f"{group_key}_{label_key}_enrichment" not in adata.uns:
        raise ValueError("Run cellcharter.gr.enrichment first.")

    if palette is None:
        palette = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", [get_cmap("coolwarm")(0), matplotlib.colors.to_rgb("darkgrey"), get_cmap("coolwarm")(255)]
        )

    enrichment = adata.uns[f"{group_key}_{label_key}_enrichment"]["enrichment"]

    if labels is not None:
        enrichment = enrichment.loc[:, labels]

    if groups is not None:
        enrichment = enrichment.loc[groups]

    size_threshold = np.max(enrichment.values) if size_threshold is None else size_threshold

    dp = _dotplot(
        adata if labels is None else adata[adata.obs[label_key].isin(labels)],
        x_key=group_key,
        y_key=label_key,
        values=enrichment,
        abs_values=False,
        size_threshold=(-1, size_threshold),
        color_threshold=(0, color_threshold),
        figsize=figsize,
        cmap=palette,
        size_title=legend_title,
        dot_scale=dot_scale,
        cluster_y=cluster_labels,
        **kwargs,
    )
    if save:
        dp.savefig(save, bbox_inches="tight")
    else:
        dp.show()
