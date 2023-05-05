from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import fowlkes_mallows_score

from cellcharter.tl import ClusterAutoK


def autok_stability(
    autok: ClusterAutoK, save: str | Path | None = None, similarity_function=fowlkes_mallows_score, n_jobs: int = 1
) -> None:
    """
    Plot the clustering stability.

    The clustering stability is computed by :class:`cellcharter.tl.ClusterAutoK`.

    Parameters
    ----------
    autok
        The fitted :class:`cellcharter.tl.ClusterAutoK` model.
    save
        Whether to save the plot.
    similarity_function
        The similarity function to use. Defaults to :func:`sklearn.metrics.fowlkes_mallows_score`.

    Returns
    -------
        Nothing, just plots the figure and optionally saves the plot.
    """
    robustness_df = pd.melt(
        pd.DataFrame.from_dict({k: autok.stability[i] for i, k in enumerate(autok.n_clusters[1:-1])}, orient="columns"),
        var_name="N. clusters",
        value_name="Stability",
    )
    ax = sns.lineplot(data=robustness_df, x="N. clusters", y="Stability")
    ax.set_xticks(autok.n_clusters[1:-1])
    if save:
        plt.savefig(save)
