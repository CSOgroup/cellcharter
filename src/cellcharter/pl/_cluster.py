from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from squidpy.gr._utils import _assert_categorical_obs
from squidpy.pl._color_utils import Palette_t, _get_palette, _maybe_set_colors

try:
    from matplotlib import colormaps as cm
except ImportError:
    from matplotlib import cm

from cellcharter.pl._utils import _dotplot


def _proportion(adata, id_key, val_key, normalize=True):
    df = pd.pivot(adata.obs[[id_key, val_key]].value_counts().reset_index(), index=id_key, columns=val_key)
    df.columns = df.columns.droplevel(0)
    if normalize:
        return df.div(df.sum(axis=1), axis=0)
    else:
        return df


def proportion(
    adata: AnnData,
    x_key: str,
    y_key: str,
    rotation_xlabel: int = 45,
    ncols=1,
    normalize=True,
    palette: Palette_t = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    return_df: bool = False,
    **kwargs,
) -> None:
    """
    Plot the proportion of `y_key` in `x_key`.

    Parameters
    ----------
    %(adata)s
    x_key
        Key in :attr:`anndata.AnnData.obs` where groups are stored.
    y_key
        Key in :attr:`anndata.AnnData.obs` where labels are stored.
    rotation_xlabel
        Rotation in degrees of the ticks of the x axis.
    ncols
        Number of panels per row.
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
    _assert_categorical_obs(adata, key=x_key)
    _assert_categorical_obs(adata, key=y_key)
    _maybe_set_colors(source=adata, target=adata, key=y_key, palette=palette)

    clusters = adata.obs[y_key].cat.categories
    palette = _get_palette(adata, cluster_key=y_key, categories=clusters)

    df = _proportion(adata=adata, id_key=x_key, val_key=y_key, normalize=normalize)
    df = df[df.columns[::-1]]

    plt.figure(dpi=dpi)
    ax = df.plot.bar(stacked=True, figsize=figsize, color=palette, rot=rotation_xlabel, ax=plt.gca(), **kwargs)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[::-1], labels[::-1], loc="center left", ncol=ncols, bbox_to_anchor=(1.0, 0.5))

    if save:
        plt.savefig(save, bbox_extra_artists=(lgd, lgd), bbox_inches="tight")
    if return_df:
        return df


def _enrichment(observed, expected, log=True):
    enrichment = observed.div(expected, axis="index", level=0)

    if log:
        enrichment = np.log2(enrichment)
    enrichment = enrichment.fillna(enrichment.min())
    return enrichment


def enrichment(
    adata: AnnData,
    group_key: str,
    label_key: str,
    size_threshold: float | None = None,
    color_threshold: float = 1,
    legend_title: str | None = None,
    dot_scale: float = 1,
    cluster_groups: bool = True,
    groups: list | None = None,
    labels: list | None = None,
    palette: Palette_t | None = None,
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
        If `True`, display groups ordered according to hierarchical clustering.
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
    if palette is None:
        palette = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", [cm.get_cmap("coolwarm")(0), matplotlib.colors.to_rgb("darkgrey"), cm.get_cmap("coolwarm")(255)]
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
        cluster_x=cluster_groups,
        **kwargs,
    )
    if save:
        dp.savefig(save, bbox_inches="tight")
    else:
        dp.show()
