from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.colors import ListedColormap
from scipy.cluster import hierarchy
from squidpy._docs import d
from squidpy.gr._utils import _assert_categorical_obs
from squidpy.pl._color_utils import Palette_t, _get_palette, _maybe_set_colors

try:
    from matplotlib.colormaps import get_cmap
except ImportError:
    from matplotlib.pyplot import get_cmap

from cellcharter.gr._group import _proportion


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
    dot_scale: float = 3,
    group_cluster: bool = True,
    label_cluster: bool = False,
    groups: list | None = None,
    labels: list | None = None,
    enriched_only: bool = True,
    palette: Palette_t | matplotlib.colors.ListedColormap | None = None,
    figsize: tuple[float, float] | None = (10, 8),
    save: str | Path | None = None,
    **kwargs,
):
    """
    Plot a dotplot of the enrichment of `label_key` in `group_key`.

    Parameters
    ----------
    %(adata)s
    group_key
        Key in :attr:`anndata.AnnData.obs` where groups are stored.
    label_key
        Key in :attr:`anndata.AnnData.obs` where labels are stored.
    dot_scale
        Scale of the dots.
    group_cluster
        If `True`, display groups ordered according to hierarchical clustering.
    label_cluster
        If `True`, display labels ordered according to hierarchical clustering.
    groups
        The groups for which to show the enrichment.
    labels
        The labels for which to show the enrichment.
    enriched_only
        If `True`, display only enriched values and hide depleted values.
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

    fold_change = adata.uns[f"{group_key}_{label_key}_enrichment"]["enrichment"].copy().T
    col_name = fold_change.columns.name
    idx_name = fold_change.index.name

    if labels is not None:
        fold_change = fold_change.loc[labels]

        # The indexing removes the name of the index, so we need to set it back
        fold_change.index.name = idx_name

    if groups is not None:
        fold_change = fold_change.loc[:, groups]

        # The indexing removes the name of the columns, so we need to set it back
        fold_change.columns.name = col_name

    if enriched_only:
        fold_change = fold_change.clip(lower=0)

    # Set -inf values to minimum and inf values to maximum
    fold_change[:] = np.nan_to_num(
        fold_change,
        neginf=np.min(fold_change[np.isfinite(fold_change)]),
        posinf=np.max(fold_change[np.isfinite(fold_change)]),
    )

    # Calculate the dendrogram for rows and columns clustering
    if label_cluster:
        order_rows = hierarchy.leaves_list(hierarchy.linkage(fold_change, method="complete"))
        fold_change = fold_change.iloc[order_rows]

    if group_cluster:
        order_cols = hierarchy.leaves_list(hierarchy.linkage(fold_change.T, method="complete"))
        fold_change = fold_change.iloc[:, order_cols]

    # Normalize the size of dots based on the absolute values in the dataframe, scaled to your preference
    clipped_sizes = np.abs(fold_change).clip(upper=np.abs(fold_change).max().max())  # .clip(upper=color_threshold[1])
    sizes = clipped_sizes * 100 / clipped_sizes.max().max() * dot_scale
    sizes = pd.melt(sizes.reset_index(), id_vars=label_key)

    labels = [0, 1]  # [01,2,3,4]
    bins = [-np.inf, 0, np.inf]  # [-np.inf, color_threshold[0], 0, color_threshold[1], np.inf]
    fold_change_color = fold_change.copy().apply(lambda x: pd.cut(x, bins=bins, labels=labels))
    fold_change_color = pd.melt(fold_change_color.reset_index(), id_vars=label_key)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Example: Blue to Red diverging palette

    # Set colormap to red if below 0, blue if above 0
    cmap = ListedColormap([cmap(0.0), cmap(1.0)])  # ListedColormap([cmap(0), cmap(90), cmap(180), cmap(360)])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")  # Set the background to white

    scatter = ax.scatter(
        pd.factorize(sizes[label_key])[0],
        pd.factorize(sizes[group_key])[0],
        s=sizes["value"],
        c=fold_change_color["value"],
        cmap=cmap,
        alpha=1,
        edgecolor="none",
        **kwargs,
    )

    handles, _ = scatter.legend_elements(
        prop="colors", num=[np.max(fold_change_color["value"]), np.min(fold_change_color["value"])]
    )

    fig.legend(handles, ["Enriched", "Depleted"], loc="outside upper left", title="", bbox_to_anchor=(0.98, 0.98))

    fig.legend(
        *scatter.legend_elements(
            prop="sizes", num=5, func=lambda x: x / 100 / dot_scale * np.abs(fold_change).max().max()
        ),
        loc="outside upper left",
        title="fold change",
        bbox_to_anchor=(0.98, 0.88),
    )

    # Adjust the ticks to match the dataframe's indices and columns
    ax.set_xticks(range(len(fold_change.index)))
    ax.set_yticks(range(len(fold_change.columns)))
    ax.set_xticklabels(fold_change.index, rotation=90)
    ax.set_yticklabels(fold_change.columns)

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight")
    else:
        plt.show()
