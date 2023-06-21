from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import _assert_categorical_obs, _assert_connectivity_key
from tqdm.auto import tqdm

from cellcharter.gr import remove_intra_cluster_links


def _observed_n_clusters_links(adj, labels, symmetric=True):
    labels_unique = labels.cat.categories
    obs = np.zeros((len(labels_unique), len(labels_unique)))
    for i, l1 in enumerate(labels_unique):
        total_cluster_links = adj[labels == l1]

        for j, l2 in enumerate(labels_unique):
            other_cluster_links = total_cluster_links[:, labels == l2]

            if not symmetric:
                obs[i, j] = np.sum(other_cluster_links) / np.sum(total_cluster_links)
            else:
                obs[i, j] = np.sum(other_cluster_links)
    obs = pd.DataFrame(obs, columns=labels_unique, index=labels_unique)
    return obs


def _expected_n_clusters_links(
    adata, clusters, cluster_key, connectivity_key=Key.obsp.spatial_conn(), only_inter=False, symmetric=True
):
    degrees = np.array([np.mean(np.sum(adata.obsp[connectivity_key], axis=1))] * adata.shape[0])

    exp = np.zeros((len(clusters), len(clusters)))
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            source_factor = np.sum(degrees[adata.obs[cluster_key] == c1])
            target_factor = np.sum(degrees[adata.obs[cluster_key] == c2])

            exp[i, j] = target_factor
            exp[i, j] /= np.sum(degrees)
            if c1 == c2:
                if symmetric:
                    exp[i, j] /= 2
                elif only_inter:
                    exp[i, j] = np.nan
            if symmetric:
                exp[i, j] *= source_factor

    exp = pd.DataFrame(exp, columns=clusters, index=clusters)
    return exp


def _observed_permuted(adj, labels, symmetric=True):
    # Permute labels
    labels = labels.sample(frac=1).reset_index(drop=True)
    return _observed_n_clusters_links(adj, labels, symmetric=symmetric)


def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: str | None = None,
    log_fold_change: bool = False,
    only_inter: bool = True,
    symmetric: bool = False,
    analytical: bool = True,
    n_perms: int = 1000,
    n_jobs: int = -1,
    copy: bool = False,
    observed_expected: bool = False,
) -> dict | None:
    """
    A modified version of squidpy's `neighborhood enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html>`_.

    This function computes the neighborhood enrichment between clusters in the spatial graph.
    It allows for the computation of the expected neighborhood enrichment using the analytical formula or by permutation.
    The analytical version is much faster, but the version based on permutation allows to estimate p-values for each enrichment value.

    Setting the symmetric parameter to `False` allows to compute the neighborhood enrichment between `cell1` and `cell2` as the ratio between the number of links between `cell1` and `cell2` and the total number of links of `cell1`.
    This results in enrichment values that are not symmetric, i.e. the neighborhood enrichment between `cell1` and `cell2` is not equal to the enrichment between `cell2` and `cell1`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(conn_key)s
    only_inter
        Consider only links between cells that belong to the different clusters.
    symmetric
        If `True`, the neighborhood enrichment between `cell1` and `cell2` is equal to the enrichment between `cell2` and `cell1`.
    analytical
        If `True`, compute the expected neighborhood enrichment using the analytical formula.
    n_perms
        Number of permutations to use to compute the expected neighborhood enrichment.
    n_jobs
        Number of jobs to run in parallel. `-1` means using all processors.
    %(copy)s
    observed_expected
        If `True`, return the observed and expected neighborhood proportions.

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the following keys:
        - ``'enrichment'`` - the neighborhood enrichment.
        - ``'pvalue'`` - the enrichment pvalues (if `analytical is False`).
        - ``'observed'`` - the observed neighborhood proportions (if `observed_expected is True`).
        - ``'expected'`` - the expected neighborhood proportions (if `observed_expected is True`).

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']`` - the above mentioned dict.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['params']`` - the parameters used.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_categorical_obs(adata, key=cluster_key)

    cluster_categories = adata.obs[cluster_key].cat.categories

    if only_inter:
        adata_copy = adata.copy()
        remove_intra_cluster_links(adata_copy, cluster_key=cluster_key)
    else:
        adata_copy = adata

    observed = _observed_n_clusters_links(
        adata_copy.obsp[connectivity_key],
        labels=adata.obs[cluster_key],
        symmetric=symmetric,
    )

    if analytical:
        expected = _expected_n_clusters_links(
            adata_copy,
            clusters=cluster_categories,
            cluster_key=cluster_key,
            connectivity_key=connectivity_key,
            only_inter=only_inter,
            symmetric=symmetric,
        )
    else:
        expected = Parallel(n_jobs=n_jobs)(
            delayed(_observed_permuted)(
                adata.obsp[connectivity_key],
                labels=adata.obs[cluster_key],
                symmetric=symmetric,
            )
            for _ in tqdm(range(n_perms))
        )

        expected = np.stack(expected, axis=0)
        pvalues = np.zeros(observed.shape)
        pvalues[observed.values > 0] = (
            1 - np.sum(expected[:, observed.values > 0] < observed.values[observed.values > 0], axis=0) / n_perms
        )
        pvalues[observed.values < 0] = (
            1 - np.sum(expected[:, observed.values < 0] > observed.values[observed.values < 0], axis=0) / n_perms
        )
        pvalues = pd.DataFrame(pvalues, columns=cluster_categories, index=cluster_categories)

        expected = np.mean(expected, axis=0)
        expected = pd.DataFrame(expected, columns=cluster_categories, index=cluster_categories)

    enrichment = np.log2(observed / expected) if log_fold_change else observed - expected

    if only_inter:
        np.fill_diagonal(observed.values, np.nan)
        np.fill_diagonal(expected.values, np.nan)
        np.fill_diagonal(enrichment.values, np.nan)

    result = {"enrichment": enrichment}

    if not analytical:
        result["pvalue"] = pvalues

    if observed_expected:
        result["observed"] = observed
        result["expected"] = expected

    if copy:
        return result
    else:
        adata.uns[f"{cluster_key}_nhood_enrichment"] = result
        adata.uns[f"{cluster_key}_nhood_enrichment"]["params"] = {
            "connectivity_key": connectivity_key,
            "log_fold_change": log_fold_change,
            "only_inter": only_inter,
            "symmetric": symmetric,
            "analytical": analytical,
            "n_perms": n_perms if not analytical else None,
        }


def diff_nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    condition_key: str,
    copy: bool = False,
    **nhood_kwargs,
) -> dict | None:
    r"""
    Differential neighborhood enrichment between conditions.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s

    condition_key
        Key in :attr:`anndata.AnnData.obs` where the sample condition is stored.

    %(copy)s
    nhood_kwargs
        Keyword arguments for :func:`gr.nhood_enrichment`.
    """
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_categorical_obs(adata, key=condition_key)

    conditions = adata.obs[condition_key].cat.categories

    if "observed_expected" in nhood_kwargs:
        warnings.warn("The `observed_expected` can be used only in `pl.nhood_enrichment`, hence it will be ignored.")

    enrichments = {}
    for condition in conditions:
        enrichments[condition] = nhood_enrichment(
            adata[adata.obs[condition_key] == condition],
            cluster_key=cluster_key,
            copy=True,
            **nhood_kwargs,
        )["enrichment"]

    result = {}
    for condition1, condition2 in combinations(conditions, 2):
        result[f"{condition1}_{condition2}"] = {"enrichment": enrichments[condition1] - enrichments[condition2]}

    if copy:
        return result
    else:
        adata.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"] = result
