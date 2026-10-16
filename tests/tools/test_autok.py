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

    @staticmethod
    def _blobs_adata():
        X, _ = make_blobs(n_samples=300, n_features=10, centers=4, cluster_std=3.0, random_state=0)
        adata = ad.AnnData(X=np.zeros((X.shape[0], 2), dtype=np.float32))
        adata.obsm["X_rep"] = X.astype(np.float32)
        return adata

    @staticmethod
    def _model_params():
        return {"random_state": 0, "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False}}

    def test_non_consecutive_n_clusters(self):
        adata = self._blobs_adata()
        autok = cc.tl.ClusterAutoK(n_clusters=[6, 2, 4], model_params=self._model_params(), max_runs=3)
        autok.fit(adata, use_rep="X_rep")

        assert autok.n_clusters == [2, 4, 6]
        # Each K is fitted together with its neighbors K-1 and K+1
        assert sorted(autok.labels.keys()) == [1, 2, 3, 4, 5, 6, 7]
        assert autok.stability.shape[0] == 3
        assert autok.best_k in autok.n_clusters
        assert len(autok.predict(adata, use_rep="X_rep").categories) == autok.best_k

    def test_tuple_equivalent_to_list(self):
        adata = self._blobs_adata()
        autok_tuple = cc.tl.ClusterAutoK(n_clusters=(2, 4), model_params=self._model_params(), max_runs=2)
        autok_list = cc.tl.ClusterAutoK(n_clusters=[2, 3, 4], model_params=self._model_params(), max_runs=2)
        autok_tuple.fit(adata, use_rep="X_rep")
        autok_list.fit(adata, use_rep="X_rep")

        assert autok_tuple.n_clusters == autok_list.n_clusters == [2, 3, 4]
        np.testing.assert_allclose(autok_tuple.stability, autok_list.stability)

    def test_n_clusters_lower_than_two_ignored(self):
        with pytest.warns(UserWarning):
            autok = cc.tl.ClusterAutoK(n_clusters=(1, 4))
        assert autok.n_clusters == [2, 3, 4]
        with pytest.raises(ValueError):
            cc.tl.ClusterAutoK(n_clusters=[1])

    def test_load_legacy_model(self):
        # Saved with n_clusters=(2, 5), when `n_clusters` also stored the neighbors 1 and 6
        autok = cc.tl.ClusterAutoK.load("tests/_models/cellcharter_autok_mibitof")
        assert autok.n_clusters == [2, 3, 4, 5]
        assert autok.best_k in autok.n_clusters
