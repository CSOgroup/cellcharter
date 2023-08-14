from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from anndata import AnnData
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import polygonize, unary_union


def _alpha_shape(coords, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Adapted from `here <https://web.archive.org/web/20200726174718/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/>`_.

    Parameters
    ----------
    coords : np.array
        Array of coordinates of points.
    alpha : float
        Alpha value to influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large, and you lose
        everything!
    Returns
    -------
    concave_hull : shapely.geometry.Polygon
        Concave hull of the points.
    """
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < alpha]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0)  # .tolist()
    m = geometry.MultiLineString(edge_points.tolist())
    triangles = list(polygonize(m.geoms))
    return unary_union(triangles), triangles, edge_points


def _process_component(points, component, hole_area_ratio=0.1, alpha_start=2000):
    alpha = alpha_start
    polygon, triangles, edge_points = _alpha_shape(points, alpha)

    while (
        type(polygon) is not geometry.polygon.Polygon
        or type(polygon) is geometry.MultiPolygon
        or edge_points.shape[0] < 10
    ):
        alpha *= 2
        polygon, triangles, edge_points = _alpha_shape(points, alpha)

    boundary_with_holes = max(triangles, key=lambda triangle: triangle.area)
    boundary = polygon

    for interior in boundary_with_holes.interiors:
        interior_polygon = geometry.Polygon(interior)
        hole_to_boundary_ratio = interior_polygon.area / boundary.area
        if hole_to_boundary_ratio > hole_area_ratio:
            try:
                difference = boundary.difference(interior_polygon)
                if isinstance(difference, geometry.Polygon):
                    boundary = difference
            except Exception:  # noqa: B902
                pass
    return component, boundary


def boundaries(
    adata: AnnData,
    cluster_key: str = "component",
    min_hole_area_ratio: float = 0.1,
    alpha_start: int = 2000,
    copy: bool = False,
) -> None | dict[int, geometry.Polygon]:
    """
    Compute the topological boundaries of sets of cells.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    min_hole_area_ratio
        Minimum ratio between the area of a hole and the area of the boundary.
    alpha_start
        Starting value for the alpha parameter of the alpha shape algorithm.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the boundaries as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['boundaries_{{cluster_key}}']`` - - the above mentioned :class:`dict`.
    """
    assert 0 <= min_hole_area_ratio <= 1, "min_hole_area_ratio must be between 0 and 1"
    assert alpha_start > 0, "alpha_start must be greater than 0"

    clusters = [cluster for cluster in adata.obs[cluster_key].unique() if cluster != -1 and not np.isnan(cluster)]

    boundaries = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _process_component,
                adata.obsm["spatial"][adata.obs[cluster_key] == cluster, :2],
                cluster,
                min_hole_area_ratio,
                alpha_start,
            ): cluster
            for cluster in clusters
        }

        for future in as_completed(futures):
            component, boundary = future.result()
            boundaries[component] = boundary

    if copy:
        return boundaries

    adata.uns[f"boundaries_{cluster_key}"] = boundaries
