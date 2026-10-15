import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sps
import squidpy as sq
from sklearn.datasets import make_blobs

import cellcharter as cc


class TestClusterAutoK:
    @pytest.mark.parametrize("dataset_name", ["mibitof"])
    def test_spatial_proteomics(self, dataset_name: str):
        download_dataset = getattr(sq.datasets, dataset_name)
        adata = download_dataset()
        if sps.issparse(adata.X):
            adata.X = adata.X.todense()
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        cc.gr.remove_long_links(adata)
        cc.gr.aggregate_neighbors(adata, n_layers=3)

        model_params = {
            "init_strategy": "kmeans",
            "random_state": 42,
            "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False},
        }
        autok = cc.tl.ClusterAutoK(
            n_clusters=(2, 5), model_class=cc.tl.GaussianMixture, model_params=model_params, max_runs=3
        )
        autok.fit(adata, use_rep="X_cellcharter")
        adata.obs[f"cellcharter_{autok.best_k}"] = autok.predict(adata, use_rep="X_cellcharter", k=autok.best_k)

        assert len(np.unique(adata.obs[f"cellcharter_{autok.best_k}"])) == autok.best_k

    @pytest.mark.parametrize("random_state", [0, -1])
    def test_random_state_reproducible(self, random_state: int):
        X, _ = make_blobs(n_samples=500, n_features=10, centers=4, cluster_std=3.0, random_state=0)
        adata = ad.AnnData(X=np.zeros((X.shape[0], 2), dtype=np.float32))
        adata.obsm["X_rep"] = X.astype(np.float32)

        def fit():
            model_params = {
                "random_state": random_state,
                "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False},
            }
            autok = cc.tl.ClusterAutoK(
                n_clusters=(2, 5), model_class=cc.tl.GaussianMixture, model_params=model_params, max_runs=2
            )
            autok.fit(adata, use_rep="X_rep")
            return autok

        autok1, autok2 = fit(), fit()
        assert autok1.best_k == autok2.best_k
        np.testing.assert_allclose(autok1.stability, autok2.stability)
