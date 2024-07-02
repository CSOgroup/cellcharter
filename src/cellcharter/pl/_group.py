from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter
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


def _legend_enrichment(scatter, enriched_only, pvalues, significance, size_threshold, dot_scale, size_max):
    handles, labels = scatter.legend_elements(prop="colors", fmt=ScalarFormatter(useMathText=False))

    handles_list = []
    labels_list = []
    empty_handle = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)

    if enriched_only is False:
        handles_list.extend(
            [
                tuple([handle for handle, label in zip(handles, labels) if int(label) in [3, 2]]),
                tuple([handle for handle, label in zip(handles, labels) if int(label) in [0, 1]]),
                empty_handle,
            ]
        )
        labels_list.extend(["Enriched", "Depleted", ""])

    if pvalues is not None:
        handles_list.extend(
            [
                tuple([handle for handle, label in zip(handles, labels) if int(label) in [0, 3]]),
                tuple([handle for handle, label in zip(handles, labels) if int(label) in [1, 2]]),
                empty_handle,
            ]
        )
        labels_list.extend([f"p-value < {significance}", f"p-value >= {significance}", ""])

    handles, labels = scatter.legend_elements(prop="sizes", num=5, func=lambda x: x / 100 / dot_scale * size_max)

    if size_threshold is not None:
        labels[-1] = f">{size_threshold:.1f}"

    handles_list.extend([empty_handle] + handles)
    labels_list.extend(["Fold change"] + labels)

    return handles_list, labels_list


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
    significance: float | None = None,
    enriched_only: bool = True,
    size_threshold: float | None = None,
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
    significance
        If not `None`, show fold changes with a p-value above this threshold in a lighter color.
    enriched_only
        If `True`, display only enriched values and hide depleted values.
    size_threshold
        Threshold for the size of the dots. Enrichment or depletions with absolute value above this threshold will have all the same size.
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
    pvalues = None
    if significance is not None:
        if "pvalue" not in adata.uns[f"{group_key}_{label_key}_enrichment"]:
            warnings.warn(
                "Significance requires gr.enrichment to be run with pvalues=True. Ignoring significance.",
                UserWarning,
                stacklevel=2,
            )
        else:
            pvalues = adata.uns[f"{group_key}_{label_key}_enrichment"]["pvalue"].copy().T

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
    sizes = np.abs(fold_change)
    size_max = sizes.max().max() if size_threshold is None else size_threshold
    if size_threshold is not None:
        sizes = sizes.clip(upper=size_threshold)

    sizes = sizes * 100 / sizes.max().max() * dot_scale
    sizes = pd.melt(sizes.reset_index(), id_vars=label_key)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Example: Blue to Red diverging palette

    # Set colormap to red if below 0, blue if above 0
    if pvalues is not None:
        fold_change_color = fold_change.copy()
        if enriched_only:
            cmap = ListedColormap([cmap(360), cmap(180)])
            fold_change_color[pvalues <= significance] = 0
            fold_change_color[pvalues > significance] = 1
        else:
            cmap = ListedColormap([cmap(0), cmap(90), cmap(180), cmap(360)])
            fold_change_color[(fold_change < 0) & (pvalues <= significance)] = 0
            fold_change_color[(fold_change < 0) & (pvalues > significance)] = 1
            fold_change_color[(fold_change >= 0) & (pvalues > significance)] = 2
            fold_change_color[(fold_change >= 0) & (pvalues <= significance)] = 3
    else:
        cmap = ListedColormap([cmap(0.0), cmap(1.0)])  # ListedColormap([cmap(0), cmap(90), cmap(180), cmap(360)])

        labels = [0, 3]
        bins = [-np.inf, 0, np.inf]
        fold_change_color = fold_change.copy().apply(lambda x: pd.cut(x, bins=bins, labels=labels))

    fold_change_color = pd.melt(fold_change_color.reset_index(), id_vars=label_key)

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

    handles, labels = _legend_enrichment(
        scatter, enriched_only, pvalues, significance, size_threshold, dot_scale, size_max
    )

    fig.legend(
        handles,
        labels,
        loc="outside upper left",
        bbox_to_anchor=(0.98, 0.98),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=1)},
        borderpad=1,
        handletextpad=1.0,
        fontsize="small",
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
