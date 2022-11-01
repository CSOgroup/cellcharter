import numpy as np
from anndata import AnnData
from squidpy._constants._pkg_constants import Key

import cellcharter as cc


class TestSpatialNeighbors:
    def test_spatial_neighbors_distance_percentile(self, non_visium_adata: AnnData):
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

        cc.gr.spatial_neighbors(non_visium_adata, dist_percentile=50)
        spatial_graph = non_visium_adata.obsp[Key.obsp.spatial_conn()].A
        spatial_dist = non_visium_adata.obsp[Key.obsp.spatial_dist()].A

        np.testing.assert_array_equal(spatial_graph, correct_graph_perc)
        np.testing.assert_allclose(spatial_dist, correct_dist_perc)
