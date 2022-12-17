from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from squidpy.gr._utils import _assert_categorical_obs
from squidpy.pl._color_utils import Palette_t, _maybe_set_colors
from squidpy.pl._graph import _get_data

from cellcharter.pl._utils import _heatmap


def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    fold_change: bool = False,
    annotate: bool = False,
    method: str | None = None,
    title: str | None = None,
    cmap: str = "bwr",
    palette: Palette_t = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
    n_digits: int = 2,
    return_enrichment: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    A modified version of squidpy's function for `plotting neighborhood enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.pl.nhood_enrichment.html>`_.

    The enrichment is computed by :func:`cellcharter.gr.nhood_enrichment`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s

    fold_change
        If `True`, the enrichment is computed as ratio between observed and expected values. Otherwise, it is computed as the difference.

    %(heatmap_plotting)s

    n_digits
        The number of digits of the number in the annotations.
    return_enrichment
        Return a :class:`pd.DataFrame` with the computed enrichment.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.text`.

    Returns
    -------
    If ``return_enrichment = True``, returns a :class:`pd.DataFrame` with the enrichment values between clusters.
    """
    _assert_categorical_obs(adata, key=cluster_key)
    nhood_enrichment_values = _get_data(adata, cluster_key=cluster_key, func_name="nhood_enrichment")
    enrichment = (
        nhood_enrichment_values["observed"] - nhood_enrichment_values["expected"]
        if not fold_change
        else nhood_enrichment_values["observed"] / nhood_enrichment_values["expected"]
    )
    adata_enrichment = AnnData(X=enrichment.astype(np.float32))
    adata_enrichment.obs[cluster_key] = pd.Categorical(enrichment.index)

    _maybe_set_colors(source=adata, target=adata_enrichment, key=cluster_key, palette=palette)
    if title is None:
        title = "Neighborhood enrichment"

    vcenter = kwargs.pop("vcenter", 1 if fold_change else 0)

    _heatmap(
        adata_enrichment,
        key=cluster_key,
        annotate=annotate,
        n_digits=n_digits,
        method=method,
        title=title,
        cont_cmap=cmap,
        figsize=(2 * adata_enrichment.n_obs // 3, 2 * adata_enrichment.n_obs // 3) if figsize is None else figsize,
        dpi=dpi,
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        vcenter=vcenter,
        **kwargs,
    )

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    if return_enrichment:
        return enrichment
