from __future__ import annotations

import numpy as np
import squidpy as sq
from anndata import AnnData
from scipy.sparse import csr_matrix
from squidpy._constants._constants import CoordType, Transform
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs


@d.dedent
@inject_docs(t=Transform, c=CoordType)
def spatial_neighbors(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_key: str | None = None,
    dist_percentile: float = 99.0,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """
    Create a graph from spatial coordinates using a modified version of Squidpy's function gr.spatial_neighbors.

    It implements an automatic procedure to remove links between cells at a distance bigger than a certain percentile of all positive distances.
    It is designed for data with generic coordinates and uses Delaunay trangulation automatically.
    For grid data like Visium, use the original function of Squidpy.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(library_key)s

    dist_percentile
        Percentile of the distances between cells over which links are trimmed after the network is built.
    set_diag
        Whether to set the diagonal of the spatial connectivities to `1.0`.
    key_added
        Key which controls where the results are saved if ``copy = False``.

    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_connectivities']`` - the spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_distances']`` - the spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}']`` - :class:`dict` containing parameters.
    """
    if copy:
        conns, dists = sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            library_key=library_key,
            delaunay=True,
            set_diag=set_diag,
            key_added=key_added,
            copy=copy,
        )
    else:
        sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            library_key=library_key,
            delaunay=True,
            set_diag=set_diag,
            key_added=key_added,
            copy=copy,
        )
        conns, dists = adata.obsp[Key.obsp.spatial_conn()], adata.obsp[Key.obsp.spatial_dist()]

    threshold = np.percentile(np.array(dists[dists != 0]).squeeze(), dist_percentile)
    conns[dists > threshold] = 0
    dists[dists > threshold] = 0

    conns.eliminate_zeros()
    dists.eliminate_zeros()

    if copy:
        return conns, dists
