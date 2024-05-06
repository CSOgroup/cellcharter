from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
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


def _expected_n_clusters_links(adj, labels, only_inter=False, symmetric=True):
    labels_unique = labels.cat.categories
    degrees = np.array([np.mean(np.sum(adj, axis=1))] * adj.shape[0])

    exp = np.zeros((len(labels_unique), len(labels_unique)))
    for i, c1 in enumerate(labels_unique):
        for j, c2 in enumerate(labels_unique):
            source_factor = np.sum(degrees[labels == c1])
            target_factor = np.sum(degrees[labels == c2])

            exp[i, j] = target_factor
            exp[i, j] /= np.sum(degrees)
            if c1 == c2:
                if symmetric:
                    exp[i, j] /= 2
                elif only_inter:
                    exp[i, j] = np.nan
            if symmetric:
                exp[i, j] *= source_factor

    exp = pd.DataFrame(exp, columns=labels_unique, index=labels_unique)
    return exp


def _empirical_pvalues(observed, expected):
    pvalues = np.zeros(observed.shape)
    pvalues[observed.values > 0] = (
        1 - np.sum(expected[:, observed.values > 0] < observed.values[observed.values > 0], axis=0) / expected.shape[0]
    )
    pvalues[observed.values < 0] = (
        1 - np.sum(expected[:, observed.values < 0] > observed.values[observed.values < 0], axis=0) / expected.shape[0]
    )
    return pd.DataFrame(pvalues, columns=observed.columns, index=observed.index)


def _observed_permuted(adj, labels, symmetric=True):
    # Permute labels
    labels = labels.sample(frac=1).reset_index(drop=True)
    return _observed_n_clusters_links(adj, labels, symmetric=symmetric)


def _nhood_enrichment(
    adj,
    labels,
    log_fold_change: bool = False,
    only_inter: bool = True,
    symmetric: bool = False,
    pvalues: bool = False,
    n_perms: int = 1000,
    n_jobs: int = -1,
    observed_expected=False,
):
    cluster_categories = labels.cat.categories

    observed = _observed_n_clusters_links(
        adj,
        labels=labels,
        symmetric=symmetric,
    )

    if not pvalues:
        expected = _expected_n_clusters_links(
            adj,
            labels=labels,
            only_inter=only_inter,
            symmetric=symmetric,
        )
    else:
        expected = Parallel(n_jobs=n_jobs)(
            delayed(_observed_permuted)(
                adj,
                labels=labels,
                symmetric=symmetric,
            )
            for _ in tqdm(range(n_perms))
        )

        expected = np.stack(expected, axis=0)

        emprical_pvalues = _empirical_pvalues(observed, expected)

        expected = np.mean(expected, axis=0)
        expected = pd.DataFrame(expected, columns=cluster_categories, index=cluster_categories)

    enrichment = np.log2(observed / expected) if log_fold_change else observed - expected
    if only_inter:
        np.fill_diagonal(observed.values, np.nan)
        np.fill_diagonal(expected.values, np.nan)
        np.fill_diagonal(enrichment.values, np.nan)

    result = {"enrichment": enrichment}

    if pvalues:
        result["pvalue"] = emprical_pvalues

    if observed_expected:
        result["observed"] = observed
        result["expected"] = expected
    return result


@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: str | None = None,
    log_fold_change: bool = False,
    only_inter: bool = True,
    symmetric: bool = False,
    pvalues: bool = False,
    n_perms: int = 1000,
    n_jobs: int = -1,
    observed_expected: bool = False,
    copy: bool = False,
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
    pvalues
        If `True`, compute the p-values for each neighborhood enrichment value using permutation of the cluster labels.
    n_perms
        Number of permutations to use to compute the expected neighborhood enrichment if `pvalues is True`.
    n_jobs
        Number of jobs to run in parallel if `pvalues is True`. `-1` means using all processors.
    %(copy)s
    observed_expected
        If `True`, return the observed and expected neighborhood proportions.

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the following keys:
        - ``'enrichment'`` - the neighborhood enrichment.
        - ``'pvalue'`` - the enrichment pvalues (if `pvalues is True`).
        - ``'observed'`` - the observed neighborhood proportions (if `observed_expected is True`).
        - ``'expected'`` - the expected neighborhood proportions (if `observed_expected is True`).

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']`` - the above mentioned dict.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['params']`` - the parameters used.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_categorical_obs(adata, key=cluster_key)

    if only_inter:
        adata_copy = adata.copy()
        remove_intra_cluster_links(adata_copy, cluster_key=cluster_key)
    else:
        adata_copy = adata

    result = _nhood_enrichment(
        adata_copy.obsp[connectivity_key],
        adata_copy.obs[cluster_key],
        log_fold_change=log_fold_change,
        only_inter=only_inter,
        symmetric=symmetric,
        pvalues=pvalues,
        n_perms=n_perms,
        n_jobs=n_jobs,
        observed_expected=observed_expected,
    )

    if copy:
        return result
    else:
        adata.uns[f"{cluster_key}_nhood_enrichment"] = result
        adata.uns[f"{cluster_key}_nhood_enrichment"]["params"] = {
            "connectivity_key": connectivity_key,
            "log_fold_change": log_fold_change,
            "only_inter": only_inter,
            "symmetric": symmetric,
            "pvalues": pvalues,
            "n_perms": n_perms if pvalues else None,
        }


def _diff_enrichment_permuted(
    adata: AnnData,
    cluster_key: str,
    subsamples_perm: np.ndarray,
    library_key: str | None = "library_id",
) -> dict:
    adata.obs["condition"] = pd.Categorical(adata.obs[library_key].isin(subsamples_perm).astype(int))
    result = diff_nhood_enrichment(adata, cluster_key=cluster_key, condition_key="condition", pvalues=False, copy=True)[
        "0_1"
    ]["enrichment"]
    return result


@d.dedent
def diff_nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    condition_key: str,
    condition_groups: tuple[str, str] | None = None,
    library_key: str | None = "library_id",
    pvalues: bool = False,
    n_perms: int = 1000,
    n_jobs: int | None = None,
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

    %(library_key)s
    pvalues
        If `True`, compute the p-values for each differential neighborhood enrichment through permutation of the condition key for each Z-dimension.
    n_perms
        Number of permutations to use to compute the expected neighborhood enrichment if `pvalues is True`.
    n_jobs
        Number of jobs to run in parallel if `pvalues is True`. `-1` means using all processors.

    %(copy)s
    nhood_kwargs
        Keyword arguments for :func:`gr.nhood_enrichment`. The following arguments are not allowed:
            - ``'observed_expected'``
            - ``n_perms``
            - ``pvalues``
            - ``n_jobs``

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` of all pairwise differential neighborhood enrichments between conditions stored as ``{condition1}_{condition2}``.
    The differential neighborhood enrichment is a :class:`dict` with the following keys:
        - ``'enrichment'`` - the differential neighborhood enrichment.
        - ``'pvalue'`` - the enrichment pvalues (if `pvalues is True`).

    """
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_categorical_obs(adata, key=condition_key)

    conditions = adata.obs[condition_key].cat.categories if condition_groups is None else condition_groups

    if "observed_expected" in nhood_kwargs:
        warnings.warn(
            "The `observed_expected` can be used only in `pl.nhood_enrichment`, hence it will be ignored.", stacklevel=2
        )

    enrichments = {}
    for condition in conditions:
        enrichments[condition] = nhood_enrichment(
            adata[adata.obs[condition_key] == condition],
            cluster_key=cluster_key,
            copy=True,
            **nhood_kwargs,
        )["enrichment"]

    result = {}

    condition_pairs = combinations(conditions, 2) if condition_groups is None else [condition_groups]

    for condition1, condition2 in condition_pairs:
        observed = enrichments[condition1] - enrichments[condition2]
        observed = observed.loc[enrichments[condition1].index, enrichments[condition1].columns]
        result_key = f"{condition1}_{condition2}"
        result[result_key] = {"enrichment": observed}
        if pvalues:
            samples1 = adata[adata.obs[condition_key] == condition1].obs[library_key].unique()
            samples2 = adata[adata.obs[condition_key] == condition2].obs[library_key].unique()

            samples_perms = np.random.choice(
                np.concatenate((samples1, samples2)), size=(n_perms, len(samples1) + len(samples2))
            )
            adata_perms = adata.copy()

            with tqdm(total=n_perms) as pbar:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = []

                    for samples_perm in samples_perms:
                        future = executor.submit(
                            _diff_enrichment_permuted,
                            adata_perms,
                            cluster_key=cluster_key,
                            subsamples_perm=samples_perm[: len(samples1)],
                            library_key=library_key,
                        )
                        futures.append(future)

                    expected = []
                    for future in as_completed(futures):
                        expected_permuted = future.result()
                        expected.append(expected_permuted)
                        pbar.update(1)

                    expected = np.stack(expected, axis=0)
                    pvalues = _empirical_pvalues(observed, expected)
                    result[result_key]["pvalue"] = pvalues

    if copy:
        return result
    else:
        adata.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"] = result
