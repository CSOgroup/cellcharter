from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy.gr._utils import _assert_categorical_obs, _assert_connectivity_key
from tqdm.auto import tqdm

from cellcharter.gr._build import _remove_intra_cluster_links


def _observed_n_clusters_links(adj, labels, symmetric=True):
    labels_unique = labels.cat.categories
    obs = np.zeros((len(labels_unique), len(labels_unique)))
    for i, l1 in enumerate(labels_unique):
        total_cluster_links = adj[labels.values == l1]

        for j, l2 in enumerate(labels_unique):
            other_cluster_links = total_cluster_links[:, labels.values == l2]

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


def _observed_permuted(adj, labels, observed, symmetric=True, batch_size=1):
    results = []
    for _ in range(batch_size):
        # Permute labels
        curr_labels = labels.sample(frac=1).reset_index(drop=True)
        permuted = _observed_n_clusters_links(adj, curr_labels, symmetric=symmetric)
        counts_pos = permuted.values[observed.values > 0] < observed.values[observed.values > 0]
        counts_neg = permuted.values[observed.values < 0] > observed.values[observed.values < 0]
        results.append((permuted, counts_pos, counts_neg))
    return results


def _nhood_enrichment(
    adj,
    labels,
    log_fold_change: bool = False,
    only_inter: bool = True,
    symmetric: bool = False,
    pvalues: bool = False,
    n_perms: int = 1000,
    n_jobs: int = 1,
    batch_size: int = 10,
    observed_expected=False,
):
    if only_inter:
        adj = _remove_intra_cluster_links(labels, adj)

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
        counts = np.zeros_like(observed.values)
        expected = np.zeros_like(observed.values)

        if n_jobs == 1:
            with tqdm(total=n_perms) as pbar:
                for _ in range(n_perms):
                    results = _observed_permuted(adj, labels=labels, observed=observed, symmetric=symmetric)
                    for permuted, counts_pos, counts_neg in results:
                        expected += permuted
                        counts[observed.values > 0] += counts_pos
                        counts[observed.values < 0] += counts_neg
                        pbar.update(1)
        else:
            n_batches = (n_perms + batch_size - 1) // batch_size

            worker = partial(
                _observed_permuted, adj, labels=labels, observed=observed, symmetric=symmetric, batch_size=batch_size
            )
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                with tqdm(total=n_perms) as pbar:
                    futures = [executor.submit(worker) for _ in range(n_batches)]

                    for future in as_completed(futures):
                        batch_results = future.result()
                        for permuted, counts_pos, counts_neg in batch_results:
                            expected += permuted
                            counts[observed.values > 0] += counts_pos
                            counts[observed.values < 0] += counts_neg
                            pbar.update(1)
                            if pbar.n >= n_perms:  # Ensure we don't exceed n_perms
                                break

        expected /= n_perms
        expected = pd.DataFrame(expected, columns=cluster_categories, index=cluster_categories)

        emprical_pvalues = pd.DataFrame(1 - (counts / n_perms), columns=observed.columns, index=observed.index)

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
    n_jobs: int = 1,
    batch_size: int = 10,
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
        Number of permutations to use to compute the expected neighborhood enrichment if `pvalues` is `True`.
    n_jobs
        Number of jobs to run in parallel if `pvalues` is `True`.
    batch_size
        Number of permutations to run in each batch if `pvalues` is `True`.
    %(copy)s
    observed_expected
        If `True`, return the observed and expected neighborhood proportions.

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the following keys:
        - ``'enrichment'`` - the neighborhood enrichment.
        - ``'pvalue'`` - the enrichment pvalues (if `pvalues` is `True`).
        - ``'observed'`` - the observed neighborhood proportions (if `observed_expected is True`).
        - ``'expected'`` - the expected neighborhood proportions (if `observed_expected is True`).

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']`` - the above mentioned dict.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['params']`` - the parameters used.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_categorical_obs(adata, key=cluster_key)

    result = _nhood_enrichment(
        adata.obsp[connectivity_key],
        adata.obs[cluster_key],
        log_fold_change=log_fold_change,
        only_inter=only_inter,
        symmetric=symmetric,
        pvalues=pvalues,
        n_perms=n_perms,
        n_jobs=n_jobs,
        batch_size=batch_size,
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


def _generate_sample_permutations(samples1, samples2, n_perms):
    """Generator function to yield sample permutations one at a time."""
    all_samples = np.concatenate((samples1, samples2))
    n_samples1 = len(samples1)

    for _ in range(n_perms):
        # Generate one permutation at a time
        perm = np.random.permutation(all_samples)
        yield perm[:n_samples1]


def _observed_expected_diff_enrichment(enrichments, condition1, condition2):
    observed = enrichments[condition1] - enrichments[condition2]
    return observed.loc[enrichments[condition1].index, enrichments[condition1].columns]


def _diff_nhood_enrichment(
    labels: pd.Categorical,
    conditions: pd.Categorical,
    condition_groups: tuple[str, str],
    connectivities: sp.csr_matrix,
    pvalues: bool = False,
    libraries: pd.Categorical | None = None,
    n_perms: int = 1000,
    n_jobs: int = 1,
    **nhood_kwargs,
):
    enrichments = {}
    for condition in condition_groups:
        condition_mask = conditions == condition
        if isinstance(condition_mask, pd.Series):
            condition_mask = condition_mask.values

        labels_condition = labels[condition_mask]
        labels_condition = labels_condition.cat.set_categories(labels.cat.categories)

        connectivities_condition = connectivities[condition_mask, :][:, condition_mask]

        enrichments[condition] = _nhood_enrichment(connectivities_condition, labels_condition, **nhood_kwargs)[
            "enrichment"
        ]

    result = {}
    condition_pairs = combinations(condition_groups, 2)

    for condition1, condition2 in condition_pairs:
        observed_diff_enrichment = _observed_expected_diff_enrichment(enrichments, condition1, condition2)
        result_key = f"{condition1}_{condition2}"
        result[result_key] = {"enrichment": observed_diff_enrichment}

        if pvalues:
            result[result_key]["pvalue_counts"] = np.zeros_like(observed_diff_enrichment.values)

            samples1 = libraries[conditions == condition1].unique()
            samples2 = libraries[conditions == condition2].unique()

            sample_perm_generator = _generate_sample_permutations(samples1, samples2, n_perms)

            if n_jobs == 1:
                with tqdm(total=n_perms) as pbar:
                    for samples_condition1_permuted in sample_perm_generator:
                        condition_permuted = pd.Categorical(libraries.isin(samples_condition1_permuted).astype(int))
                        expected_diff_enrichment = _diff_nhood_enrichment(
                            labels, condition_permuted, [0, 1], libraries, connectivities, pvalues=False, **nhood_kwargs
                        )["0_1"]["enrichment"]

                        counts_pos = (
                            expected_diff_enrichment.values[observed_diff_enrichment.values > 0]
                            < observed_diff_enrichment.values[observed_diff_enrichment.values > 0]
                        )
                        counts_neg = (
                            expected_diff_enrichment.values[observed_diff_enrichment.values < 0]
                            > observed_diff_enrichment.values[observed_diff_enrichment.values < 0]
                        )

                        result[result_key]["pvalue_counts"][observed_diff_enrichment.values > 0] += counts_pos
                        result[result_key]["pvalue_counts"][observed_diff_enrichment.values < 0] += counts_neg
                        pbar.update(1)
            else:
                worker = partial(
                    _diff_nhood_enrichment,
                    labels=labels,
                    condition_groups=[0, 1],
                    libraries=libraries,
                    connectivities=connectivities,
                    pvalues=False,
                    **nhood_kwargs,
                )
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    with tqdm(total=n_perms) as pbar:
                        futures = []
                        for _ in range(n_perms):
                            condition_permuted = pd.Categorical(libraries.isin(next(sample_perm_generator)).astype(int))
                            futures.append(executor.submit(worker, conditions=condition_permuted))

                        for future in as_completed(futures):
                            future_result = future.result()
                            expected_diff_enrichment = future_result["0_1"]["enrichment"]

                            counts_pos = (
                                expected_diff_enrichment.values[observed_diff_enrichment.values > 0]
                                < observed_diff_enrichment.values[observed_diff_enrichment.values > 0]
                            )
                            counts_neg = (
                                expected_diff_enrichment.values[observed_diff_enrichment.values < 0]
                                > observed_diff_enrichment.values[observed_diff_enrichment.values < 0]
                            )

                            result[result_key]["pvalue_counts"][observed_diff_enrichment.values > 0] += counts_pos
                            result[result_key]["pvalue_counts"][observed_diff_enrichment.values < 0] += counts_neg
                            pbar.update(1)
    return result


@d.dedent
def diff_nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    condition_key: str,
    condition_groups: tuple[str, str] | None = None,
    connectivity_key: str | None = None,
    pvalues: bool = False,
    library_key: str | None = "library_id",
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
    condition_groups
        The condition groups to compare. If `None`, all conditions in `condition_key` will be used.
    %(conn_key)s
    pvalues
        If `True`, compute the p-values for each differential neighborhood enrichment through permutation of the condition key for each Z-dimension.
    library_key
        If multiple library_id, column in anndata.AnnData.obs which stores mapping between library_id and obs. Used only if `pvalues` is `True` to permute the condition labels.
    n_perms
        Number of permutations to use to compute the expected neighborhood enrichment if `pvalues` is `True`.
    n_jobs
        Number of jobs to run in parallel if `pvalues` is `True`. `-1` means using all processors.

    %(copy)s
    nhood_kwargs
        Keyword arguments for :func:`gr.nhood_enrichment`. The following arguments are not allowed:
            - ``n_perms``
            - ``pvalues``
            - ``n_jobs``

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` of all pairwise differential neighborhood enrichments between conditions stored as ``{condition1}_{condition2}``.
    The differential neighborhood enrichment is a :class:`dict` with the following keys:
        - ``'enrichment'`` - the differential neighborhood enrichment.
        - ``'pvalue'`` - the enrichment pvalues (if `pvalues` is `True`).

    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_categorical_obs(adata, key=condition_key)

    if condition_groups is None:
        condition_groups = adata.obs[condition_key].cat.categories

    if "observed_expected" in nhood_kwargs:
        warnings.warn(
            "The `observed_expected` can be used only in `pl.nhood_enrichment`, hence it will be ignored.", stacklevel=2
        )

    diff_nhood = _diff_nhood_enrichment(
        labels=adata.obs[cluster_key],
        conditions=adata.obs[condition_key],
        condition_groups=condition_groups,
        connectivities=adata.obsp[connectivity_key],
        pvalues=pvalues,
        libraries=adata.obs[library_key] if pvalues else None,
        n_perms=n_perms,
        n_jobs=n_jobs,
        **nhood_kwargs,
    )

    result = {}
    for condition_pair_key in diff_nhood.keys():
        result[condition_pair_key] = {}
        result[condition_pair_key]["enrichment"] = diff_nhood[condition_pair_key]["enrichment"]
        if pvalues:
            result[condition_pair_key]["pvalue"] = pd.DataFrame(
                1 - (diff_nhood[condition_pair_key]["pvalue_counts"] / n_perms),
                columns=diff_nhood[condition_pair_key]["enrichment"].columns,
                index=diff_nhood[condition_pair_key]["enrichment"].index,
            )

    if copy:
        return result
    else:
        adata.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"] = result
