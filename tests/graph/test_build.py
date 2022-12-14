import numpy as np
import scipy.sparse as sps
import squidpy as sq
from anndata import AnnData
from squidpy._constants._pkg_constants import Key

import cellcharter as cc


class TestRemoveLongLinks:
    def test_remove_long_links(self, non_visium_adata: AnnData):
        # ground-truth removing connections longer that 50th percentile
        correct_dist_perc = np.array(
            [
                [0.0, 2.0, 0.0, 4.12310563],
                [2.0, 0.0, 0, 5.0],
                [0.0, 0, 0.0, 0.0],
                [4.12310563, 5.0, 0.0, 0.0],
            ]
        )
        correct_graph_perc = np.array(
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
        )

        sq.gr.spatial_neighbors(non_visium_adata, coord_type="generic", delaunay=True)
        cc.gr.remove_long_links(non_visium_adata, distance_percentile=50)

        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        spatial_dist = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(spatial_graph, correct_graph_perc)
        np.testing.assert_allclose(spatial_dist, correct_dist_perc)


class TestRemoveIntraClusterLinks:
    def test_mixed_clusters(self, non_visium_adata: AnnData):
        non_visium_adata.obsp[Key.obsp.spatial_conn()] = sps.csr_matrix(
            np.ones((non_visium_adata.shape[0], non_visium_adata.shape[0]))
        )
        non_visium_adata.obsp[Key.obsp.spatial_dist()] = sps.csr_matrix(
            [[0, 1, 4, 4], [1, 0, 6, 3], [4, 6, 0, 9], [4, 3, 9, 0]]
        )
        non_visium_adata.obs["cluster"] = [0, 0, 1, 1]

        correct_conns = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
        correct_dists = np.array([[0, 0, 4, 4], [0, 0, 6, 3], [4, 6, 0, 0], [4, 3, 0, 0]])

        cc.gr.remove_intra_cluster_links(non_visium_adata, cluster_key="cluster")

        trimmed_conns = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        trimmed_dists = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(trimmed_conns, correct_conns)
        np.testing.assert_allclose(trimmed_dists, correct_dists)

    def test_same_clusters(self, non_visium_adata: AnnData):
        non_visium_adata.obsp[Key.obsp.spatial_conn()] = sps.csr_matrix(
            np.ones((non_visium_adata.shape[0], non_visium_adata.shape[0]))
        )
        non_visium_adata.obsp[Key.obsp.spatial_dist()] = sps.csr_matrix(
            [[0, 1, 4, 4], [1, 0, 6, 3], [4, 6, 0, 9], [4, 3, 9, 0]]
        )
        non_visium_adata.obs["cluster"] = [0, 0, 0, 0]

        correct_conns = np.zeros((non_visium_adata.shape[0], non_visium_adata.shape[0]))
        correct_dists = np.zeros((non_visium_adata.shape[0], non_visium_adata.shape[0]))

        cc.gr.remove_intra_cluster_links(non_visium_adata, cluster_key="cluster")

        trimmed_conns = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        trimmed_dists = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(trimmed_conns, correct_conns)
        np.testing.assert_allclose(trimmed_dists, correct_dists)

    def test_different_clusters(self, non_visium_adata: AnnData):
        non_visium_adata.obsp[Key.obsp.spatial_conn()] = sps.csr_matrix(
            np.ones((non_visium_adata.shape[0], non_visium_adata.shape[0]))
        )
        non_visium_adata.obsp[Key.obsp.spatial_dist()] = sps.csr_matrix(
            [[0, 1, 4, 4], [1, 0, 6, 3], [4, 6, 0, 9], [4, 3, 9, 0]]
        )
        non_visium_adata.obs["cluster"] = [0, 1, 2, 3]

        correct_conns = non_visium_adata.obsp[Key.obsp.spatial_conn()].copy()
        correct_conns.setdiag(0)
        correct_dists = non_visium_adata.obsp[Key.obsp.spatial_dist()]

        cc.gr.remove_intra_cluster_links(non_visium_adata, cluster_key="cluster")

        trimmed_conns = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        trimmed_dists = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(trimmed_conns, correct_conns.A)
        np.testing.assert_allclose(trimmed_dists, correct_dists.A)

    def test_copy(self, non_visium_adata: AnnData):
        non_visium_adata.obsp[Key.obsp.spatial_conn()] = sps.csr_matrix(
            np.ones((non_visium_adata.shape[0], non_visium_adata.shape[0]))
        )
        non_visium_adata.obsp[Key.obsp.spatial_dist()] = sps.csr_matrix(
            [[0, 1, 4, 4], [1, 0, 6, 3], [4, 6, 0, 9], [4, 3, 9, 0]]
        )
        non_visium_adata.obs["cluster"] = [0, 0, 1, 1]

        correct_conns = non_visium_adata.obsp[Key.obsp.spatial_conn()].copy()
        correct_dists = non_visium_adata.obsp[Key.obsp.spatial_dist()].copy()

        cc.gr.remove_intra_cluster_links(non_visium_adata, cluster_key="cluster", copy=True)

        trimmed_conns = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        trimmed_dists = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(trimmed_conns, correct_conns.A)
        np.testing.assert_allclose(trimmed_dists, correct_dists.A)
