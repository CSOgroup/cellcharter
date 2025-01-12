from __future__ import annotations

import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np
import shapely
import sknw
from anndata import AnnData
from matplotlib.path import Path
from rasterio import features
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import polygonize, unary_union
from skimage.morphology import skeletonize
from squidpy._docs import d


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


@d.dedent
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
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['boundaries']`` - the above mentioned :class:`dict`.
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

    adata.uns[f"shape_{cluster_key}"] = {"boundary": boundaries}


def _find_dangling_branches(graph, total_length, min_ratio=0.05):
    total_length = np.sum(list(nx.get_edge_attributes(graph, "weight").values()))
    adj = nx.to_numpy_array(graph, weight=None)
    adj_w = nx.to_numpy_array(graph)

    n_neighbors = np.sum(adj, axis=1)
    node_total_dist = np.sum(adj_w, axis=1)
    dangling_nodes = np.argwhere((node_total_dist < min_ratio * total_length) & (n_neighbors == 1))
    if dangling_nodes.shape[0] != 1:
        dangling_nodes = dangling_nodes.squeeze()
    else:
        dangling_nodes = dangling_nodes[0]
    return dangling_nodes


def _remove_dangling_branches(graph, min_ratio=0.05):
    total_length = np.sum(list(nx.get_edge_attributes(graph, "weight").values()))

    dangling_branches = _find_dangling_branches(graph, total_length=total_length, min_ratio=min_ratio)

    while len(dangling_branches) > 0:
        idx2node = dict(enumerate(graph.nodes))
        for i in dangling_branches:
            graph.remove_node(idx2node[i])

        dangling_branches = _find_dangling_branches(graph, total_length=total_length, min_ratio=min_ratio)


def _longest_path_from_node(graph, u):
    visited = dict.fromkeys(graph.nodes)
    distance = {i: -1 for i in list(graph.nodes)}
    idx2node = dict(enumerate(graph.nodes))

    try:
        adj_lil = nx.to_scipy_sparse_matrix(graph, format="lil")
    except AttributeError:
        adj_lil = nx.to_scipy_sparse_array(graph, format="lil")
    adj = {i: [idx2node[neigh] for neigh in neighs] for i, neighs in zip(graph.nodes, adj_lil.rows)}
    weight = nx.get_edge_attributes(graph, "weight")

    distance[u] = 0
    queue = deque()
    queue.append(u)
    visited[u] = True
    while queue:
        front = queue.popleft()
        for i in adj[front]:
            if not visited[i]:
                visited[i] = True
                source, target = min(i, front), max(i, front)
                distance[i] = distance[front] + weight[(source, target)]
                queue.append(i)

    farthest_node = max(distance, key=distance.get)

    longest_path_length = distance[farthest_node]
    return farthest_node, longest_path_length


def _longest_path_length(graph):
    # first DFS to find one end point of longest path
    node, _ = _longest_path_from_node(graph, list(graph.nodes)[0])
    # second DFS to find the actual longest path
    _, longest_path_length = _longest_path_from_node(graph, node)
    return longest_path_length


def _linearity(boundary, height=1000, min_ratio=0.05):
    img, _ = _rasterize(boundary, height=height)
    skeleton = skeletonize(img).astype(int)

    graph = sknw.build_sknw(skeleton.astype(np.uint16))
    graph = graph.to_undirected()

    _remove_dangling_branches(graph, min_ratio=min_ratio)

    cycles = nx.cycle_basis(graph)
    cycles_len = [nx.path_weight(graph, cycle + [cycle[0]], "weight") for cycle in cycles]

    longest_path_length = _longest_path_length(graph)
    longest_length = np.max(cycles_len + [longest_path_length])

    return longest_length / np.sum(list(nx.get_edge_attributes(graph, "weight").values()))


def _rasterize(boundary, height=1000):
    minx, miny, maxx, maxy = boundary.bounds
    poly = shapely.affinity.translate(boundary, -minx, -miny)
    if maxx - minx > maxy - miny:
        scale_factor = height / poly.bounds[2]
    else:
        scale_factor = height / poly.bounds[3]
    poly = shapely.affinity.scale(poly, scale_factor, scale_factor, origin=(0, 0, 0))
    return features.rasterize([poly], out_shape=(height, int(height * (maxx - minx) / (maxy - miny)))), scale_factor


def linearity(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "linearity",
    height: int = 1000,
    min_ratio: float = 0.05,
    copy: bool = False,
) -> None | dict[int, float]:
    warnings.warn(
        "linearity is deprecated and will be removed in the next release. " "Please use `linearity_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return linearity_metric(
        adata=adata,
        cluster_key=cluster_key,
        out_key=out_key,
        height=height,
        min_ratio=min_ratio,
        copy=copy,
    )


@d.dedent
def linearity_metric(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "linearity",
    height: int = 1000,
    min_ratio: float = 0.05,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the linearity of the topological boundaries of sets of cells.

    It rasterizes the polygon and computes the skeleton of the rasterized image.
    Then, it computes the longest path in the skeleton and divides it by the total length of the skeleton.
    Branches that are shorter than ``min_ratio`` times the total length of the skeleton are removed because are not considered real branches.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    height
        Height of the rasterized image. The width is computed automatically to preserve the aspect ratio of the polygon. Higher values lead to more precise results but also higher memory usage.
    min_ratio
        Minimum ratio between the length of a branch and the total length of the skeleton to be considered a real branch and not be removed.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the linearity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    linearity_score = {}
    for cluster, boundary in boundaries.items():
        linearity_score[cluster] = _linearity(boundary, height=height, min_ratio=min_ratio)

    if copy:
        return linearity_score

    adata.uns[f"shape_{cluster_key}"][out_key] = linearity_score


def _elongation(boundary):
    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*boundary.minimum_rotated_rectangle.exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)
    return 1 - minor_axis / major_axis


def elongation(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "elongation",
    copy: bool = False,
) -> None | dict[int, float]:
    warnings.warn(
        "elongation is deprecated and will be removed in the next release. " "Please use `elongation_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return elongation_metric(
        adata=adata,
        cluster_key=cluster_key,
        out_key=out_key,
        copy=copy,
    )


@d.dedent
def elongation_metric(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "elongation",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the elongation of the topological boundaries of sets of cells.

    It computes the minimum bounding rectangle of the polygon and divides the length of the minor axis by the length of the major axis.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the elongation as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    elongation_score = {}
    for cluster, boundary in boundaries.items():
        elongation_score[cluster] = _elongation(boundary)

    if copy:
        return elongation_score
    adata.uns[f"shape_{cluster_key}"][out_key] = elongation_score


def _axes(boundary):
    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*boundary.minimum_rotated_rectangle.exterior.coords.xy))
    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]
    return min(mbr_lengths), max(mbr_lengths)


def _curl(boundary):
    factor = boundary.length**2 - 16 * boundary.area
    if factor < 0:
        factor = 0
    fibre_length = boundary.area / ((boundary.length - np.sqrt(factor)) / 4)

    _, length = _axes(boundary)
    if fibre_length < length:
        return 0
    else:
        return 1 - length / fibre_length


def curl(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "curl",
    copy: bool = False,
) -> None | dict[int, float]:
    warnings.warn(
        "curl is deprecated and will be removed in the next release. " "Please use `curl_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return curl_metric(
        adata=adata,
        cluster_key=cluster_key,
        out_key=out_key,
        copy=copy,
    )


@d.dedent
def curl_metric(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "curl",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the curl score of the topological boundaries of sets of cells.

    It computes the curl score of each cluster as one minues the ratio between the length of the major axis of the minimum bounding rectangle and the fiber length of the polygon.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the curl score as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.

    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]
    curl_score = {}
    for cluster, boundary in boundaries.items():
        curl_score[cluster] = _curl(boundary)

    if copy:
        return curl_score
    adata.uns[f"shape_{cluster_key}"][out_key] = curl_score


def purity_metric(
    adata: AnnData,
    cluster_key: str = "component",
    library_key: str = "sample",
    out_key: str = "purity",
    exterior: bool = False,
    copy: bool = False,
) -> None | dict[int, float]:
    warnings.warn(
        "purity is deprecated and will be removed in the next release. " "Please use `purity_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return purity(
        adata=adata,
        cluster_key=cluster_key,
        library_key=library_key,
        out_key=out_key,
        exterior=exterior,
        copy=copy,
    )


@d.dedent
def purity(
    adata: AnnData,
    cluster_key: str = "component",
    library_key: str = "sample",
    out_key: str = "purity",
    exterior: bool = False,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the purity of the topological boundaries of sets of cells.

    It computes the purity of each cluster as the ratio between the number of cells of the cluster that are within the boundary and the total number of cells within the boundary.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    exterior
        If ``True``, the computation of the purity ignores the polygon's internal holes.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the purity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    purity_score = {}
    for cluster, boundary in boundaries.items():
        sample = adata[adata.obs[cluster_key] == cluster].obs[library_key][0]
        adata_sample = adata[adata.obs[library_key] == sample]

        points = adata_sample.obsm["spatial"][:, :2]
        within_mask = np.zeros(points.shape[0], dtype=bool)
        if type(boundary) is geometry.multipolygon.MultiPolygon:
            for p in boundary.geoms:
                path = Path(np.array(p.exterior.coords.xy).T)
                within_mask |= np.array(path.contains_points(points))
        else:
            path = Path(np.array(boundary.exterior.coords.xy).T)
            within_mask |= np.array(path.contains_points(points))
            if not exterior:
                for interior in boundary.interiors:
                    path = Path(np.array(interior.coords.xy).T)
                    within_mask &= ~np.array(path.contains_points(points))

        purity_score[cluster] = np.sum(adata_sample.obs[cluster_key][within_mask] == cluster) / np.sum(within_mask)

    if copy:
        return purity_score
    adata.uns[f"shape_{cluster_key}"][out_key] = purity_score
