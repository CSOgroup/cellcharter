from __future__ import annotations

import anndata as ad
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from spatialdata_plot.pl.utils import _get_colors_for_categorical_obs


def plot_boundaries(
    adata: AnnData,
    sample: str,
    library_key: str = "sample",
    cluster_key: str = "component",
    alpha_boundary: float = 0.5,
    show_cells: bool = True,
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
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    alpha_boundary
        Transparency of the boundaries.
    show_cells
        Whether to show the cells or not.

    Returns
    -------
    %(plotting_returns)s
    """
    adata = adata[adata.obs[library_key] == sample].copy()
    del adata.raw
    clusters = adata.obs[cluster_key].unique()

    boundaries = {
        cluster: boundary for cluster, boundary in adata.uns[f"boundaries_{cluster_key}"].items() if cluster in clusters
    }
    gdf = geopandas.GeoDataFrame(geometry=list(boundaries.values()))
    # adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key])
    adata.obs.loc[adata.obs[cluster_key] == -1, cluster_key] = np.nan
    adata.obs.index = "cell_" + adata.obs.index
    adata.obs["instance_id"] = adata.obs.index
    adata.obs["region"] = "cells"

    xy = adata.obsm["spatial"]
    cell_circles = sd.models.ShapesModel.parse(xy, geometry=0, radius=3000, index=adata.obs["instance_id"])

    obs = pd.DataFrame(list(boundaries.keys()), columns=[cluster_key], index=np.arange(len(boundaries)).astype(str))
    adata_obs = ad.AnnData(X=pd.DataFrame(index=obs.index, columns=adata.var_names), obs=obs)
    adata_obs.obs["region"] = "clusters"
    adata_obs.index = "cell_" + adata_obs.obs.index
    adata_obs.obs["instance_id"] = adata_obs.obs.index
    adata_obs.obs[cluster_key] = pd.Categorical(adata_obs.obs[cluster_key])

    adata = ad.concat((adata, adata_obs), join="outer")

    adata.obs["region"] = adata.obs["region"].astype("category")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    table = sd.models.TableModel.parse(
        adata, region_key="region", region=["clusters", "cells"], instance_key="instance_id"
    )

    shapes = {
        "clusters": sd.models.ShapesModel.parse(gdf),
        "cells": sd.models.ShapesModel.parse(cell_circles),
    }
    adata.obs["instance_id"] = pd.Categorical(adata.obs["instance_id"])

    sdata = sd.SpatialData(shapes=shapes, table=table)

    ax = plt.gca()
    if show_cells:
        sdata.pl.render_shapes(elements="cells", color=cluster_key).pl.show(ax=ax, legend_loc=False)
    else:
        sdata.table.uns[f"{cluster_key}_colors"] = _get_colors_for_categorical_obs(
            adata.obs["component"].cat.categories
        )

    sdata.pl.render_shapes(
        elements="clusters",
        color=cluster_key,
        fill_alpha=alpha_boundary,
        palette=mpl.colors.ListedColormap(sdata.table.uns[f"{cluster_key}_colors"]),
    ).pl.show(ax=ax)
    plt.show()
