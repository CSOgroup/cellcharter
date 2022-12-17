from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import _assert_categorical_obs, _assert_connectivity_key

import cellcharter as cc


def _observed_n_clusters_links(adata, clusters, cluster_key, connectivity_key=Key.obsp.spatial_conn(), symmetric=True):
    obs = np.zeros((len(clusters), len(clusters)))
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            total_cluster_links = adata.obsp[connectivity_key][adata.obs[cluster_key] == c1]
            other_cluster_links = total_cluster_links[:, adata.obs[cluster_key] == c2]

            if not symmetric:
                obs[i, j] = np.sum(other_cluster_links) / np.sum(total_cluster_links)
            else:
                obs[i, j] = np.sum(other_cluster_links)
    obs = pd.DataFrame(obs, columns=clusters, index=clusters)
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


def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: str | None = None,
    only_inter: bool = True,
    symmetric: bool = False,
    copy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    A modified version of squidpy's `neighborhood enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html>`_ computed analytically and with the possibility of asymmetric enrichment.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(conn_key)s

    only_inter
        Consider only links between cells that belong to the different clusters.

    symmetric
        If `True`, the neighborhood enrichment between `cell1` and `cell2` is equal to the enrichment between `cell2` and `cell1`.

    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the observed and expected neighborhood enrichment.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['observed']`` - the observed enrichment.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['expected']`` - the expected enrichment.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_categorical_obs(adata, key=cluster_key)

    cluster_categories = adata.obs[cluster_key].cat.categories

    if only_inter:
        adata_copy = adata.copy()
        cc.gr.remove_intra_cluster_links(adata_copy, cluster_key=cluster_key)
    else:
        adata_copy = adata

    observed = _observed_n_clusters_links(
        adata_copy,
        clusters=cluster_categories,
        cluster_key=cluster_key,
        connectivity_key=connectivity_key,
        symmetric=symmetric,
    )
    expected = _expected_n_clusters_links(
        adata_copy,
        clusters=cluster_categories,
        cluster_key=cluster_key,
        connectivity_key=connectivity_key,
        only_inter=only_inter,
        symmetric=symmetric,
    )

    if copy:
        return observed, expected
    else:
        adata.uns[f"{cluster_key}_nhood_enrichment"] = {"observed": observed, "expected": expected}


# def _nhood_enrichment_diff(adata,
#     cluster_key,
#     group_key,
#     group1,
#     group2,
#     connectivity_key=Key.obsp.spatial_conn(),
#     filter_clusters=None,
#     only_inter=True,
#     symmetric=False,
#     cache=None,
# ):
#     observed, expected = nhood_enrichment(
#         adata=adata[adata.obs[group_key].isin(group1)],
#         cluster_key=cluster_key,
#         connectivity_key=connectivity_key,
#         filter_clusters=filter_clusters,
#         only_inter=only_inter,
#         symmetric=symmetric,
#         cache=cache,
#         cache_group=group1
#     )
#     group1_enrichment = observed - expected

#     observed, expected = nhood_enrichment(
#         adata=adata[adata.obs[group_key].isin(group2)],
#         cluster_key=cluster_key,
#         connectivity_key=connectivity_key,
#         filter_clusters=filter_clusters,
#         only_inter=only_inter,
#         symmetric=symmetric,
#         cache=cache,
#         cache_group=group2
#     )

#     group2_enrichment = observed - expected
#     return group1_enrichment - group2_enrichment


# def nhood_enrichment_diff(
#     adata,
#     cluster_key,
#     group_key,
#     group1,
#     group2=None,
#     connectivity_key=Key.obsp.spatial_conn(),
#     filter_clusters=None,
#     only_inter=True,
#     symmetric=False,
#     pval=None,
#     n_perms=None,
#     use_cache=True
# ):
#     group1 = [group1] if isinstance(group1, str) else group1
#     if group2 is None:
#         group2 = [label for label in adata.obs[group_key].cat.categories if label not in group1]
#     else:
#         group2 = [group2] if isinstance(group2, str) else group2
#     n_observed = len(group1)
#     n_expected = len(group2)

#     observed_diff = _nhood_enrichment_diff(
#         adata=adata,
#             cluster_key=cluster_key,
#             group_key=group_key,
#             group1=group1,
#             group2=group2,
#             connectivity_key=connectivity_key,
#             filter_clusters=filter_clusters,
#             only_inter=only_inter,
#             symmetric=symmetric,
#             cache=None
#     )

#     cache = dict() if use_cache else None
#     expected_diffs = list()
#     for samples_perm in tqdm(np.random.choice(list(group1)+list(group2), size=(n_perms, n_observed+n_expected))):

#         expected_diff = _nhood_enrichment_diff(
#             adata=adata,
#             cluster_key=cluster_key,
#             group_key=group_key,
#             group1=samples_perm[:n_observed],
#             group2=samples_perm[n_observed:],
#             connectivity_key=connectivity_key,
#             filter_clusters=filter_clusters,
#             only_inter=only_inter,
#             symmetric=symmetric,
#             cache=cache
#         )

#         expected_diffs.append(expected_diff)

#     expected_diffs = np.stack(expected_diffs, axis=0)

#     empirical_pvals = np.zeros(observed_diff.shape)
#     empirical_pvals[observed_diff.values > 0] = 1 - np.sum(expected_diffs[:, observed_diff.values > 0] < observed_diff.values[observed_diff.values > 0], axis=0) / n_perms
#     empirical_pvals[observed_diff.values < 0] = 1 - np.sum(expected_diffs[:, observed_diff.values < 0] > observed_diff.values[observed_diff.values < 0], axis=0) / n_perms

#     observed_diff[empirical_pvals > pval] = np.nan

#     return observed_diff, empirical_pvals
