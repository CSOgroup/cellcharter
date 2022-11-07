import numpy as np
import scipy.sparse as sps
import squidpy as sq
from anndata import AnnData

from cellcharter.gr import aggregate_neighbors


class TestAggregateNeighbors:
    def test_aggregate_neighbors(self):
        n_layers = 2
        aggregations = ["mean", "var"]

        G = sps.csr_matrix(
            np.array(
                [
                    [0, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 1, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                ]
            )
        )

        X = np.vstack((np.power(2, np.arange(G.shape[0])), np.power(2, np.arange(G.shape[0])[::-1]))).T.astype(
            np.float32
        )

        adata = AnnData(X=X, obsp={"spatial_connectivities": G})

        L1_mean_truth = np.vstack(
            ([7.5, 16.5, 64.66, 91, 4.5, 2, 66, 34, 8], [60, 132, 87.33, 91, 144, 128, 33, 34, 32])
        ).T.astype(np.float32)

        L2_mean_truth = np.vstack(
            ([120, 9.33, 8.67, 3, 87.33, 1, 1, 1, 8.5], [3.75, 37.33, 58.67, 96, 64.33, 256, 256, 256, 136])
        ).T.astype(np.float32)

        L1_var_truth = np.vstack(
            ([5.36, 15.5, 51.85, 116.83, 3.5, 0, 62, 30, 0], [42.90, 124, 119.27, 116.83, 112, 0, 31, 30, 0])
        ).T.astype(np.float32)

        L2_var_truth = np.vstack(
            ([85.79, 4.99, 5.73, 1.0, 119.27, 0, 0, 0, 7.5], [2.68, 19.96, 49.46, 32, 51.85, 0, 0, 0, 120])
        ).T.astype(np.float32)

        aggregate_neighbors(adata, n_layers=n_layers, aggregations=["mean", "var"])

        assert adata.obsm["X_cellcharter"].shape == (adata.shape[0], X.shape[1] * (n_layers * len(aggregations) + 1))

        np.testing.assert_allclose(adata.obsm["X_cellcharter"][:, [0, 1]], X, rtol=0.01)
        np.testing.assert_allclose(adata.obsm["X_cellcharter"][:, [2, 3]], L1_mean_truth, rtol=0.01)
        np.testing.assert_allclose(adata.obsm["X_cellcharter"][:, [4, 5]], L1_var_truth**2, rtol=0.01)
        np.testing.assert_allclose(adata.obsm["X_cellcharter"][:, [6, 7]], L2_mean_truth, rtol=0.01)
        np.testing.assert_allclose(adata.obsm["X_cellcharter"][:, [8, 9]], L2_var_truth**2, rtol=0.01)

    def test_aggregations(self, adata: AnnData):
        sq.gr.spatial_neighbors(adata)
        aggregate_neighbors(adata, n_layers=3, aggregations="mean", out_key="X_str")
        aggregate_neighbors(adata, n_layers=3, aggregations=["mean"], out_key="X_list")

        assert (adata.obsm["X_str"] != adata.obsm["X_list"]).nnz == 0
