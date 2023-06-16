from __future__ import annotations

import warnings
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
    annotate: bool = False,
    method: str | None = None,
    title: str | None = "Neighborhood enrichment",
    cmap: str = "bwr",
    palette: Palette_t = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
    n_digits: int = 2,
    significance: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    A modified version of squidpy's function for `plotting neighborhood enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.pl.nhood_enrichment.html>`_.

    The enrichment is computed by :func:`cellcharter.gr.nhood_enrichment`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(heatmap_plotting)s

    n_digits
        The number of digits of the number in the annotations.
    significance
        Mark the values that are below this threshold with a star. If `None`, no significance is computed. It requires ``gr.nhood_enrichment`` to be run with ``analytical=False``.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.text`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    nhood_enrichment_values = _get_data(adata, cluster_key=cluster_key, func_name="nhood_enrichment")
    enrichment = nhood_enrichment_values["enrichment"]
    enrichment[np.isinf(enrichment)] = np.nan
    adata_enrichment = AnnData(X=enrichment.astype(np.float32))
    adata_enrichment.obs[cluster_key] = pd.Categorical(enrichment.index)

    if significance is not None:
        if "pvalue" not in nhood_enrichment_values:
            warnings.warn(
                "Significance requires gr.nhood_enrichment to be run with analytical=False. Ignoring significance.",
                UserWarning,
            )
        else:
            adata_enrichment.layers["significant"] = np.empty_like(enrichment, dtype=str)
            adata_enrichment.layers["significant"][nhood_enrichment_values["pvalue"] <= significance] = "*"

    _maybe_set_colors(source=adata, target=adata_enrichment, key=cluster_key, palette=palette)

    vcenter = kwargs.pop("vcenter", 1 if nhood_enrichment_values["params"]["log_fold_change"] else 0)

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
