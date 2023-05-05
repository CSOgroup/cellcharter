from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from squidpy._constants._constants import CoordType, Transform
from squidpy._constants._pkg_constants import Key
from squidpy._docs import inject_docs
from squidpy.gr._utils import _assert_connectivity_key


@inject_docs(t=Transform, c=CoordType)
def remove_long_links(
    adata: AnnData,
    distance_percentile: float = 99.0,
    connectivity_key: str | None = None,
    distances_key: str | None = None,
    neighs_key: str | None = None,
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """
    Remove links between cells at a distance bigger than a certain percentile of all positive distances.

    It is designed for data with generic coordinates.

    Parameters
    ----------
    %(adata)s

    distance_percentile
        Percentile of the distances between cells over which links are trimmed after the network is built.
    %(conn_key)s

    distances_key
        Key in :attr:`anndata.AnnData.obsp` where spatial distances are stored.
        Default is: :attr:`anndata.AnnData.obsp` ``['{{Key.obsp.spatial_dist()}}']``.
    neighs_key
        Key in :attr:`anndata.AnnData.uns` where the parameters from gr.spatial_neighbors are stored.
        Default is: :attr:`anndata.AnnData.uns` ``['{{Key.uns.spatial_neighs()}}']``.

    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the new spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:
        - :attr:`anndata.AnnData.obsp` ``['{{connectivity_key}}']`` - the new spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{distances_key}}']`` - the new spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{neighs_key}}']`` - :class:`dict` containing parameters.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    distances_key = Key.obsp.spatial_dist(distances_key)
    neighs_key = Key.uns.spatial_neighs(neighs_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_connectivity_key(adata, distances_key)

    conns, dists = adata.obsp[connectivity_key], adata.obsp[distances_key]

    if copy:
        conns, dists = conns.copy(), dists.copy()

    threshold = np.percentile(np.array(dists[dists != 0]).squeeze(), distance_percentile)
    conns[dists > threshold] = 0
    dists[dists > threshold] = 0

    conns.eliminate_zeros()
    dists.eliminate_zeros()

    if copy:
        return conns, dists
    else:
        adata.uns[neighs_key]["params"]["radius"] = threshold


def remove_intra_cluster_links(
    adata: AnnData,
    cluster_key: str,
    connectivity_key: str | None = None,
    distances_key: str | None = None,
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """
    Remove links between cells that belong to the same cluster.

    Used in :func:`cellcharter.gr.nhood_enrichment` to consider only interactions between cells of different clusters.

    Parameters
    ----------
    %(adata)s

    cluster_key
        Key in :attr:`anndata.AnnData.obs` of the cluster labeling to consider.

    %(conn_key)s

    distances_key
        Key in :attr:`anndata.AnnData.obsp` where spatial distances are stored.
        Default is: :attr:`anndata.AnnData.obsp` ``['{{Key.obsp.spatial_dist()}}']``.

    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the new spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:
        - :attr:`anndata.AnnData.obsp` ``['{{connectivity_key}}']`` - the new spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{distances_key}}']`` - the new spatial distances.
    """
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    distances_key = Key.obsp.spatial_dist(distances_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_connectivity_key(adata, distances_key)

    conns = adata.obsp[connectivity_key].copy() if copy else adata.obsp[connectivity_key]
    dists = adata.obsp[distances_key].copy() if copy else adata.obsp[distances_key]

    for matrix in [conns, dists]:
        target_clusters = np.array(adata.obs[cluster_key][matrix.indices])
        source_clusters = np.array(
            adata.obs[cluster_key][np.repeat(np.arange(matrix.indptr.shape[0] - 1), np.diff(matrix.indptr))]
        )

        inter_cluster_mask = (source_clusters != target_clusters).astype(int)

        matrix.data *= inter_cluster_mask
        matrix.eliminate_zeros()

    if copy:
        return conns, dists
