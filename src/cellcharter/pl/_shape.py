from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    cell_radius: float = 1.0,
    save: str | Path | None = None,
    **kwargs,
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
    cell_radius
        Radius of the cells, if present.
    save
        Path to save the plot.
    kwargs
        Additional arguments to pass to the `spatialdata.pl.render_shapes()` function.

    Returns
    -------
    %(plotting_returns)s
    """
    adata = adata[adata.obs[library_key] == sample].copy()
    del adata.raw
    clusters = adata.obs[component_key].unique()

    boundaries = {
        cluster: boundary
        for cluster, boundary in adata.uns[f"shape_{component_key}"]["boundary"].items()
        if cluster in clusters
    }
    gdf = geopandas.GeoDataFrame(geometry=list(boundaries.values()), index=np.arange(len(boundaries)).astype(str))

    adata_components = ad.AnnData(
        obs=pd.DataFrame(list(boundaries.keys()), columns=[component_key], index=np.arange(len(boundaries)).astype(str))
    )
    adata_components.obs["region"] = "component_boundaries"
    adata_components.obs["region"] = pd.Categorical(adata_components.obs["region"])
    adata_components.index = "cluster_" + adata_components.obs.index
    adata_components.obs["instance_id"] = np.arange(len(boundaries)).astype(str)
    adata_components.obs[component_key] = pd.Categorical(adata_components.obs[component_key])
    adata_components.obs[component_key] = adata_components.obs[component_key].cat.remove_unused_categories()

    shapes = {"component_boundaries": sd.models.ShapesModel.parse(gdf)}

    tables = {
        "components": sd.models.TableModel.parse(
            adata_components, region_key="region", region="component_boundaries", instance_key="instance_id"
        )
    }

    if show_cells:
        adata_cells = ad.AnnData(obs=adata.obs[[component_key]], obsm={"spatial": adata.obsm["spatial"]})
        adata_cells.obs.loc[adata_cells.obs[component_key] == -1, component_key] = np.nan
        adata_cells.obs.index = "cell_" + adata_cells.obs.index
        adata_cells.obs["instance_id"] = adata_cells.obs.index
        adata_cells.obs["region"] = "cell_circles"
        adata_cells.obs["region"] = pd.Categorical(adata_cells.obs["region"])

        # Ensure component_key is categorical
        if not pd.api.types.is_categorical_dtype(adata_cells.obs[component_key]):
            adata_cells.obs[component_key] = pd.Categorical(adata_cells.obs[component_key])

        # Check if spatial data exists
        if "spatial" not in adata_cells.obsm or adata_cells.obsm["spatial"].shape[0] == 0:
            warnings.warn("No spatial data found for cells. Skipping cell visualization.", stacklevel=2)
            show_cells = False
        else:
            tables["cells"] = sd.models.TableModel.parse(
                adata_cells, region_key="region", region="cell_circles", instance_key="instance_id"
            )
            shapes["cell_circles"] = sd.models.ShapesModel.parse(
                adata_cells.obsm["spatial"], geometry=0, radius=1.0, index=adata_cells.obs["instance_id"]
            )

    sdata = sd.SpatialData(shapes=shapes, tables=tables)

    _, ax = plt.subplots(**kwargs)

    palette = None
    groups = None
    if show_cells:
        try:
            if pd.api.types.is_categorical_dtype(sdata.tables["cells"].obs[component_key]):
                groups = list(sdata.tables["cells"].obs[component_key].cat.categories)
            else:
                groups = list(sdata.tables["cells"].obs[component_key].unique())
            # Remove any NaN values from groups
            groups = [g for g in groups if pd.notna(g)]
        except (KeyError, AttributeError) as e:
            warnings.warn(f"Could not determine groups for plotting: {e}", stacklevel=2)
            groups = None

        from squidpy.pl._color_utils import _maybe_set_colors

        _maybe_set_colors(
            source=sdata.tables["cells"], target=sdata.tables["cells"], key=component_key, palette=palette
        )
        palette = sdata.tables["cells"].uns[f"{component_key}_colors"]

        try:
            sdata.pl.render_shapes(
                "cell_circles",
                color=component_key,
                scale=cell_radius,
                palette=palette,
                groups=groups,
                method="matplotlib",
            ).pl.show(ax=ax, legend_loc=None)
        except TypeError:  # TODO: remove after spatialdata-plot issue  #256 is fixed
            warnings.warn(
                "Until the next spatialdata_plot release, the cells that do not belong to any component will be displayed with a random color instead of grey.",
                stacklevel=2,
            )
            # Create a copy of the table with modified component labels
            modified_table = sdata.tables["cells"].copy()
            modified_table.obs[component_key] = modified_table.obs[component_key].cat.add_categories([-1])
            modified_table.obs[component_key] = modified_table.obs[component_key].fillna(-1)

            # Update the spatialdata object with the modified table
            sdata.tables["cells"] = modified_table

            sdata.pl.render_shapes(
                "cell_circles",
                color=component_key,
                scale=cell_radius,
                palette=palette,
                groups=groups,
                method="matplotlib",
            ).pl.show(ax=ax, legend_loc=None)

    sdata.pl.render_shapes(
        element="component_boundaries",
        color=component_key,
        fill_alpha=alpha_boundary,
        palette=palette,
        groups=groups if groups is not None else list(adata_components.obs[component_key].cat.categories),
        method="matplotlib",
    ).pl.show(ax=ax)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")


def plot_shape_metrics(
    adata: AnnData,
    condition_key: str,
    condition_groups: list[str] | None = None,
    cluster_key: str | None = None,
    cluster_groups: list[str] | None = None,
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
        cluster_groups=cluster_groups,
        component_key=component_key,
        metrics=metrics,
        figsize=figsize,
        title=title,
    )


def plot_shapes(data, x, y, hue, hue_order, fig, ax, fontsize: str | int = 14, title: str | None = None) -> None:
    """
    Create a boxplot with stripplot overlay for shape metrics visualization.

    Parameters
    ----------
    data
        DataFrame containing the data to plot.
    x
        Column name for x-axis variable.
    y
        Column name for y-axis variable.
    hue
        Column name for grouping variable.
    hue_order
        Order of hue categories.
    fig
        Matplotlib figure object.
    ax
        Matplotlib axes object.
    fontsize
        Font size for plot elements.
    title
        Title for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """
    new_ax = sns.boxplot(data=data, x=x, hue=hue, y=y, showfliers=False, hue_order=hue_order, ax=ax)
    adjust_box_widths(fig, 0.9)

    new_ax = sns.stripplot(
        data=data,
        x=x,
        hue=hue,
        y=y,
        palette="dark:0.08",
        size=4,
        jitter=0.13,
        dodge=True,
        hue_order=hue_order,
        ax=new_ax,
    )

    if len(data[hue].unique()) > 1:
        handles, labels = new_ax.get_legend_handles_labels()
        if len(handles) > 1:
            new_ax.legend(
                handles[0 : len(data[hue].unique())],
                labels[0 : len(data[hue].unique())],
                bbox_to_anchor=(1.0, 1.03),
                title=hue,
                prop={"size": fontsize},
                title_fontsize=fontsize,
            )
    else:
        if new_ax.get_legend() is not None:
            new_ax.get_legend().remove()

    new_ax.set_ylim(-0.05, 1.05)
    new_ax.set_title(title, fontdict={"fontsize": fontsize})
    new_ax.tick_params(axis="both", labelsize=fontsize)
    new_ax.set_xlabel(new_ax.get_xlabel(), fontsize=fontsize)
    new_ax.set_ylabel(new_ax.get_ylabel(), fontsize=fontsize)
    return new_ax


@d.dedent
def shape_metrics(
    adata: AnnData,
    condition_key: str | None = None,
    condition_groups: list[str] | None = None,
    cluster_key: str | None = None,
    cluster_groups: str | list[str] | None = None,
    component_key: str = "component",
    metrics: str | tuple[str] | list[str] | None = None,
    fontsize: str | int = 14,
    figsize: tuple[float, float] = (10, 7),
    ncols: int = 2,
) -> None:
    """
    Boxplots of the shape metrics between two conditions.

    Parameters
    ----------
    %(adata)s
    condition_key
        Key in :attr:`anndata.AnnData.obs` where the condition labels are stored.
    condition_groups
        List of conditions to show. If None, all conditions are plotted.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored. This is used to filter the clusters to plot.
    cluster_groups
        List of cluster to plot. If None, all clusters are plotted.
    component_key
        Key in :attr:`anndata.AnnData.obs` where the component labels are stored.
    metrics
        List of metrics to plot. Available metrics are ``linearity``, ``curl``, ``elongation``, ``purity``, ``rcs``. If `None`, all computed metrics are plotted.
    figsize
        Figure size.
    ncols
        Number of columns in the subplot grid when plotting multiple metrics.
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

    if (
        cluster_groups is not None
        and not isinstance(cluster_groups, list)
        and not isinstance(cluster_groups, np.ndarray)
    ):
        cluster_groups = [cluster_groups]

    if condition_groups is None and condition_key is not None:
        condition_groups = adata.obs[condition_key].cat.categories
    else:
        if not isinstance(condition_groups, list) and not isinstance(condition_groups, np.ndarray):
            condition_groups = [condition_groups]

    if metrics is None:
        metrics = [
            metric
            for metric in ["linearity", "curl", "elongation", "purity", "rcs"]
            if metric in adata.uns[f"shape_{component_key}"]
        ]

    keys = []
    if condition_key is not None:
        keys.append(condition_key)
    if cluster_key is not None:
        keys.append(cluster_key)

    metrics_df = adata.obs[[component_key] + keys].drop_duplicates().dropna().set_index(component_key)

    for metric in metrics:
        metrics_df[metric] = metrics_df.index.map(adata.uns[f"shape_{component_key}"][metric])

    if cluster_groups is not None:
        metrics_df = metrics_df[metrics_df[cluster_key].isin(cluster_groups)]

        metrics_melted = pd.melt(
            metrics_df,
            id_vars=keys,
            value_vars=metrics,
            var_name="metric",
        )

        metrics_melted[cluster_key] = metrics_melted[cluster_key].cat.remove_unused_categories()

        if cluster_key is not None and condition_key is not None and metrics_melted[condition_key].nunique() >= 2:
            nrows = (2 + ncols - 1) // ncols  # Ceiling division
            # Create figure with appropriate size
            fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            # Calculate average axes height in inches
            avg_height = figsize[1] / 2
            # Set absolute spacing of 1.5 inches between subplots
            fig.subplots_adjust(hspace=1.5 / avg_height)

            plot_shapes(
                metrics_melted,
                "metric",
                "value",
                cluster_key,
                cluster_groups,
                fig=fig,
                ax=axes[0],
                title="Shape metrics by domain",
                fontsize=fontsize,
            )

            plot_shapes(
                metrics_melted,
                "metric",
                "value",
                condition_key,
                condition_groups,
                fig=fig,
                ax=axes[1],
                title="Shape metrics by condition",
                fontsize=fontsize,
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            if cluster_key is not None:
                plot_shapes(
                    metrics_melted,
                    "metric",
                    "value",
                    cluster_key,
                    cluster_groups,
                    fig=fig,
                    ax=ax,
                    title="Shape metrics by domain",
                    fontsize=fontsize,
                )

            if condition_key is not None:
                if metrics_melted[condition_key].nunique() < 2:
                    warnings.warn(
                        f"Only one condition {condition_groups[0]} for domain {cluster_groups}. Skipping condition plot.",
                        stacklevel=2,
                    )
                else:
                    plot_shapes(
                        metrics_melted,
                        "metric",
                        "value",
                        condition_key,
                        condition_groups,
                        fig=fig,
                        ax=ax,
                        title="Shape metrics by condition",
                        fontsize=fontsize,
                    )
    else:
        # Calculate number of rows needed based on number of metrics and ncols
        n_metrics = len(metrics)
        nrows = (n_metrics + ncols - 1) // ncols  # Ceiling division

        # Create figure with appropriate size
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each metric in its own subplot
        for i, metric in enumerate(metrics):
            ax = axes[i]
            plot_shapes(
                metrics_df,
                cluster_key if cluster_key is not None else condition_key,
                metric,
                condition_key if condition_key is not None else cluster_key,
                condition_groups if condition_groups is not None else None,
                fig=fig,
                ax=ax,
                title=f"Spatial domains: {metric}",
                fontsize=fontsize,
            )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
