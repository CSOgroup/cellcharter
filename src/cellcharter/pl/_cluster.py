from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData
from squidpy._docs import d
from squidpy.gr._utils import _assert_categorical_obs
from squidpy.pl._color_utils import Palette_t, _get_palette, _maybe_set_colors


def _proportion(adata, id_key, val_key, normalize=True):
    df = pd.pivot(adata.obs[[id_key, val_key]].value_counts().reset_index(), index=id_key, columns=val_key)
    df.columns = df.columns.droplevel(0)
    if normalize:
        return df.div(df.sum(axis=1), axis=0)
    else:

        return df


@d.dedent
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
