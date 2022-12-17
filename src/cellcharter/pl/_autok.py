from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cellcharter.tl import ClusterAutoK


def autok_stability(autok: ClusterAutoK, save: str | Path | None = None) -> None:
    """
    Plot the clustering stability.

    The clustering stability is computed by :class:`cellcharter.tl.ClusterAutoK`.

    Parameters
    ----------
    autok
        The fitted :class:`cellcharter.tl.ClusterAutoK` model.
    save
        Whether to save the plot.

    Returns
    -------
        Nothing, just plots the figure and optionally saves the plot.
    """
    n_clusters = list(autok.stability.keys())[1:-1]
    robustness_df = pd.melt(
        pd.DataFrame.from_dict({k: autok.stability[k] for k in n_clusters}, orient="columns"),
        var_name="N. clusters",
        value_name="Fowlkes-Mallows score",
    )
    ax = sns.lineplot(data=robustness_df, x="N. clusters", y="Fowlkes-Mallows score")
    ax.set_xticks(n_clusters)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.clf()
