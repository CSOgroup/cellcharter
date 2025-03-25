from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import anndata as ad
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
import seaborn as sns
import spatialdata as sd
import spatialdata_plot  # noqa: F401
from anndata import AnnData
from squidpy._docs import d

from ._utils import adjust_box_widths


def plot_boundaries(
    adata: AnnData,
    sample: str,
    library_key: str = "sample",
    component_key: str = "component",
    alpha_boundary: float = 0.5,
    show_cells: bool = True,
    save: str | Path | None = None,
) -> None:
    """
    Plot the boundaries of the clusters.

    Parameters
    ----------
    %(adata)s
    sample
        Sample to plot.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored.
    component_key
        Key in :attr:`anndata.AnnData.obs` where the component labels are stored.
    alpha_boundary
        Transparency of the boundaries.
    show_cells
        Whether to show the cells or not.

    Returns
    -------
    %(plotting_returns)s
    """
    # Print warning and call boundaries
    warnings.warn(
        "plot_boundaries is deprecated and will be removed in the next release. " "Please use `boundaries` instead.",
        FutureWarning,
        stacklevel=2,
    )
    boundaries(
        adata=adata,
        sample=sample,
        library_key=library_key,
        component_key=component_key,
        alpha_boundary=alpha_boundary,
        show_cells=show_cells,
        save=save,
    )


@d.dedent
def boundaries(
    adata: AnnData,
    sample: str,
    library_key: str = "sample",
    component_key: str = "component",
    alpha_boundary: float = 0.5,
    show_cells: bool = True,
    cells_radius: float = None,
    save: str | Path | None = None,
) -> None:
    """
    Plot the boundaries of the clusters.

    Parameters
    ----------
    %(adata)s
    sample
        Sample to plot.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored.
    component_key
        Key in :attr:`anndata.AnnData.obs` where the component labels are stored.
    alpha_boundary
        Transparency of the boundaries.
    show_cells
        Whether to show the cells or not.

    Returns
    -------
    %(plotting_returns)s
    """
    if show_cells is True and cells_radius is None:
        raise ValueError("cells_radius must be provided when show_cells is True")

    adata = adata[adata.obs[library_key] == sample].copy()
    del adata.raw
    clusters = adata.obs[component_key].unique()

    boundaries = {
        cluster: boundary
        for cluster, boundary in adata.uns[f"shape_{component_key}"]["boundary"].items()
        if cluster in clusters
    }
    gdf = geopandas.GeoDataFrame(geometry=list(boundaries.values()))
    adata.obs.loc[adata.obs[component_key] == -1, component_key] = np.nan
    adata.obs.index = "cell_" + adata.obs.index
    adata.obs["instance_id"] = adata.obs.index
    adata.obs["region"] = "cells"

    xy = adata.obsm["spatial"]
    cell_circles = sd.models.ShapesModel.parse(xy, geometry=0, radius=cells_radius, index=adata.obs["instance_id"])

    obs = pd.DataFrame(list(boundaries.keys()), columns=[component_key], index=np.arange(len(boundaries)).astype(str))
    adata_obs = ad.AnnData(X=pd.DataFrame(index=obs.index, columns=adata.var_names), obs=obs)
    adata_obs.obs["region"] = "clusters"
    adata_obs.index = "cluster_" + adata_obs.obs.index
    adata_obs.obs["instance_id"] = np.arange(len(boundaries))
    adata_obs.obs[component_key] = pd.Categorical(adata_obs.obs[component_key])

    if sps.issparse(adata.X):
        # If the adata is sparse, we need to convert the adata_obs to an empty sparse matrix
        adata_obs.X = sps.csr_matrix((len(adata_obs.obs), len(adata.var_names)))

    adata = ad.concat((adata, adata_obs), join="outer")

    adata.obs["region"] = adata.obs["region"].astype("category")

    table = sd.models.TableModel.parse(
        adata, region_key="region", region=["clusters", "cells"], instance_key="instance_id"
    )

    shapes = {
        "clusters": sd.models.ShapesModel.parse(gdf),
        "cells": sd.models.ShapesModel.parse(cell_circles),
    }

    sdata = sd.SpatialData(shapes=shapes, tables=table)

    ax = plt.gca()
    if show_cells:
        try:
            sdata.pl.render_shapes(elements="cells", color=component_key).pl.show(ax=ax, legend_loc=None)
        except TypeError:  # TODO: remove after spatialdata-plot issue  #256 is fixed
            warnings.warn(
                "Until the next spatialdata_plot release, the cells that do not belong to any component will be displayed with a random color instead of grey.",
                stacklevel=2,
            )
            sdata.tables["table"].obs[component_key] = sdata.tables["table"].obs[component_key].cat.add_categories([-1])
            sdata.tables["table"].obs[component_key] = sdata.tables["table"].obs[component_key].fillna(-1)
            sdata.pl.render_shapes(elements="cells", color=component_key).pl.show(ax=ax, legend_loc=None)

    sdata.pl.render_shapes(
        element="clusters",
        color=component_key,
        fill_alpha=alpha_boundary,
    ).pl.show(ax=ax)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")


def plot_shape_metrics(
    adata: AnnData,
    condition_key: str,
    condition_groups: list[str] | None = None,
    cluster_key: str | None = None,
    cluster_id: list[str] | None = None,
    component_key: str = "component",
    metrics: str | tuple[str] | list[str] = ("linearity", "curl"),
    figsize: tuple[float, float] = (8, 7),
    title: str | None = None,
) -> None:
    """
    Boxplots of the shape metrics between two conditions.

    Parameters
    ----------
    %(adata)s
    condition_key
        Key in :attr:`anndata.AnnData.obs` where the condition labels are stored.
    condition_groups
        List of two conditions to compare. If None, all pairwise comparisons are made.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored. This is used to filter the clusters to plot.
    cluster_id
        List of clusters to plot. If None, all clusters are plotted.
    component_key
        Key in :attr:`anndata.AnnData.obs` where the component labels are stored.
    metrics
        List of metrics to plot. Available metrics are ``linearity``, ``curl``, ``elongation``, ``purity``.
    figsize
        Figure size.
    title
        Title of the plot.

    Returns
    -------
    %(plotting_returns)s
    """
    # Print warning and call shape_metrics
    warnings.warn(
        "plot_shape_metrics is deprecated and will be removed in the next release. "
        "Please use `shape_metrics` instead.",
        FutureWarning,
        stacklevel=2,
    )
    shape_metrics(
        adata=adata,
        condition_key=condition_key,
        condition_groups=condition_groups,
        cluster_key=cluster_key,
        cluster_id=cluster_id,
        component_key=component_key,
        metrics=metrics,
        figsize=figsize,
        title=title,
    )


def plot_shapes(data, x, y, hue, hue_order, figsize, title: str | None = None) -> None:
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=data,
        x=x,
        hue=hue,
        y=y,
        showfliers=False,
        hue_order=hue_order,
    )
    adjust_box_widths(fig, 0.9)

    ax = sns.stripplot(
        data=data,
        x=x,
        hue=hue,
        y=y,
        palette="dark:0.08",
        size=4,
        jitter=0.13,
        dodge=True,
        hue_order=hue_order,
    )

    if len(data[hue].unique()) > 1:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:
            plt.legend(
                handles[0 : len(data[hue].unique())],
                labels[0 : len(data[hue].unique())],
                bbox_to_anchor=(1.0, 1.03),
                title=hue,
            )
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.show()


@d.dedent
def shape_metrics(
    adata: AnnData,
    condition_key: str | None = None,
    condition_groups: list[str] | None = None,
    cluster_key: str | None = None,
    cluster_id: str | list[str] | None = None,
    component_key: str = "component",
    metrics: str | tuple[str] | list[str] | None = None,
    fontsize: str | int = "small",
    figsize: tuple[float, float] = (8, 7),
    title: str | None = None,
) -> None:
    """
    Boxplots of the shape metrics between two conditions.

    Parameters
    ----------
    %(adata)s
    condition_key
        Key in :attr:`anndata.AnnData.obs` where the condition labels are stored.
    condition_groups
        List of two conditions to compare. If None, all pairwise comparisons are made.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored. This is used to filter the clusters to plot.
    cluster_id
        List of clusters to plot. If None, all clusters are plotted.
    component_key
        Key in :attr:`anndata.AnnData.obs` where the component labels are stored.
    metrics
        List of metrics to plot. Available metrics are ``linearity``, ``curl``, ``elongation``, ``purity``, ``ncc``. If `None`, all computed metrics are plotted.
    figsize
        Figure size.
    title
        Title of the plot.

    Returns
    -------
    %(plotting_returns)s
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    elif isinstance(metrics, tuple):
        metrics = list(metrics)

    if cluster_id is not None and not isinstance(cluster_id, list) and not isinstance(cluster_id, np.ndarray):
        cluster_id = [cluster_id]

    if condition_groups is None and condition_key is not None:
        condition_groups = adata.obs[condition_key].cat.categories
    else:
        if not isinstance(condition_groups, list) and not isinstance(condition_groups, np.ndarray):
            condition_groups = [condition_groups]

    if metrics is None:
        metrics = [metric for metric in adata.uns[f"shape_{component_key}"].keys() if metric != "boundary"]

    keys = []
    if condition_key is not None:
        keys.append(condition_key)
    if cluster_key is not None:
        keys.append(cluster_key)

    metrics_df = adata.obs[[component_key] + keys].drop_duplicates().dropna().set_index(component_key)

    for metric in metrics:
        metrics_df[metric] = metrics_df.index.map(adata.uns[f"shape_{component_key}"][metric])

    if cluster_id is not None:
        metrics_df = metrics_df[metrics_df[cluster_key].isin(cluster_id)]

        metrics_melted = pd.melt(
            metrics_df,
            id_vars=keys,
            value_vars=metrics,
            var_name="metric",
        )

        metrics_melted[cluster_key] = metrics_melted[cluster_key].cat.remove_unused_categories()

        if cluster_key is not None:
            plot_shapes(
                metrics_melted,
                "metric",
                "value",
                cluster_key,
                cluster_id,
                figsize,
                f'Spatial domains: {", ".join([str(cluster) for cluster in cluster_id])} by domain',
            )
            plot_shapes(
                metrics_melted,
                "metric",
                "value",
                cluster_key,
                cluster_id,
                figsize,
                f'Spatial domains: {", ".join([str(cluster) for cluster in cluster_id])}',
            )

        if condition_key is not None:
            plot_shapes(
                metrics_melted,
                "metric",
                "value",
                condition_key,
                condition_groups,
                figsize,
                f'Spatial domains: {", ".join([str(cluster) for cluster in cluster_id])} by condition',
            )
            plot_shapes(
                metrics_melted,
                "metric",
                "value",
                condition_key,
                condition_groups,
                figsize,
                f'Spatial domains: {", ".join([str(cluster) for cluster in cluster_id])}',
            )
    else:
        for metric in metrics:
            plot_shapes(
                metrics_df,
                cluster_key if cluster_key is not None else condition_key,
                metric,
                condition_key if condition_key is not None else cluster_key,
                condition_groups if condition_groups is not None else None,
                figsize,
                f"Spatial domains: {metric}",
            )
