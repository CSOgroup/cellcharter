from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sps
from anndata import AnnData
from scipy.sparse import spdiags
from squidpy._constants._pkg_constants import Key as sqKey
from tqdm.auto import tqdm

from cellcharter._constants._pkg_constants import Key
from cellcharter._utils import str2list


def _aggregate_mean(adj, x):
    return adj @ x


def _aggregate_var(adj, x):
    mean = adj @ x
    mean_squared = adj @ (x * x)
    return mean_squared - mean * mean


def _aggregate(adj, x, method):
    if method == "mean":
        return _aggregate_mean(adj, x)
    elif method == "var":
        return _aggregate_var(adj, x)
    else:
        raise NotImplementedError


def _mul_broadcast(mat1, mat2):
    return spdiags(mat2, 0, len(mat2), len(mat2)) * mat1


def _hop(adj_hop, adj, adj_visited=None):
    adj_hop = adj_hop @ adj

    if adj_visited is not None:
        adj_hop = adj_hop > adj_visited  # Logical not for sparse matrices
        adj_visited = adj_visited + adj_hop

    return adj_hop, adj_visited


def _normalize(adj):
    deg = np.array(np.sum(adj, axis=1)).squeeze()

    with warnings.catch_warnings():
        # If a cell doesn't have neighbors deg = 0 -> divide by zero
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0

    return _mul_broadcast(adj, deg_inv)


def _setdiag(array, value):
    if isinstance(array, sps.csr_matrix):
        array = array.tolil()
    array.setdiag(value)
    array = array.tocsr()
    if value == 0:
        array.eliminate_zeros()
    return array


def _aggregate_neighbors(
    adj: sps.spmatrix,
    X: np.ndarray,
    nhood_layers: list,
    aggregations: Optional[Union[str, list]] = "mean",
    disable_tqdm: bool = True,
) -> np.ndarray:

    adj = adj.astype(bool)
    adj = _setdiag(adj, 0)
    adj_hop = adj.copy()
    adj_visited = _setdiag(adj.copy(), 1)

    Xs = []
    for i in tqdm(range(0, max(nhood_layers) + 1), disable=disable_tqdm):

        if i in nhood_layers:
            if i == 0:
                Xs.append(X)
            else:
                if i > 1:
                    adj_hop, adj_visited = _hop(adj_hop, adj, adj_visited)
                adj_hop_norm = _normalize(adj_hop)

                for agg in aggregations:
                    x = _aggregate(adj_hop_norm, X, agg)
                    Xs.append(x)
    if sps.issparse(X):
        return sps.hstack(Xs)
    else:
        return np.hstack(Xs)


def aggregate_neighbors(
    adata: AnnData,
    n_layers: Union[int, list],
    aggregations: Optional[Union[str, list]] = "mean",
    connectivity_key: Optional[str] = None,
    use_rep: Optional[str] = None,
    sample_key: Optional[str] = None,
    out_key: Optional[str] = "X_cellcharter",
    copy: bool = False,
) -> np.ndarray | None:
    """
    Aggregate the features from each neighborhood layers and concatenate them, and optionally with the cells' features, into a single vector.

    Parameters
    ----------
    %(adata)s
    n_layers
        Which neighborhood layers to aggregate from.
        If :class:`int`, the output vector includes the cells' features and the aggregated features of the neighbors until the layer at distance ``n_layers``, i.e. cells | 1-hop neighbors | ... | ``n_layers``-hop.
        If :class:`list`, every element corresponds to the distances at which the neighbors' features will be aggregated and concatenated. For example, [0, 1, 3] corresponds to cells|1-hop neighbors|3-hop neighbors.
    aggregations
        Which functions to use to aggregate the neighbors features. Default: ```mean``.
    connectivity_key
        Key in :attr:`anndata.AnnData.obsp` where spatial connectivities are stored.
    use_rep
        Key of the features. If :class:`None`, adata.X is used. Else, the key is used to access the field in the AnnData .obsm mapping.
    sample_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored. Must be different from :class:`None` if adata contains multiple samples.
    out_key
        Key in :attr:`anndata.AnnData.obsm` where the output matrix is stored if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`numpy.ndarray` of the features aggregated and concatenated.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.obsm` ``['{{out_key}}']`` - the above mentioned :class:`numpy.ndarray`.
    """
    connectivity_key = sqKey.obsp.spatial_conn(connectivity_key)
    sample_key = Key.obs.sample if sample_key is None else sample_key
    aggregations = str2list(aggregations)

    X = adata.X if use_rep is None else adata.obsm[use_rep]

    if isinstance(n_layers, int):
        n_layers = list(range(n_layers + 1))

    if sps.issparse(X):
        X_aggregated = sps.dok_matrix(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)), dtype=np.float32
        )
    else:
        X_aggregated = np.empty(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)), dtype=np.float32
        )

    if sample_key in adata.obs:
        samples = adata.obs[sample_key].unique()
        sample_idxs = [adata.obs[sample_key] == sample for sample in samples]
    else:
        sample_idxs = [np.arange(adata.shape[0])]

    for idxs in tqdm(sample_idxs, disable=(len(sample_idxs) == 1)):
        X_sample_aggregated = _aggregate_neighbors(
            adj=adata[idxs].obsp[connectivity_key],
            X=X[idxs],
            nhood_layers=n_layers,
            aggregations=aggregations,
            disable_tqdm=(len(sample_idxs) != 1),
        )
        X_aggregated[idxs] = X_sample_aggregated

    if isinstance(X_aggregated, sps.dok_matrix):
        X_aggregated = X_aggregated.tocsr()

    if copy:
        return X_aggregated

    adata.obsm[out_key] = X_aggregated
