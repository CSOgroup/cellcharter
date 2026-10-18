import pickle
from shutil import copytree

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
        """Fit and predict spatial clusters on the proteomics reference dataset."""
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
        """Reusing a seed reproduces the selected K and stability values."""
        X, _ = make_blobs(n_samples=500, n_features=10, centers=4, cluster_std=3.0, random_state=0)
        adata = ad.AnnData(X=np.zeros((X.shape[0], 2), dtype=np.float32))
        adata.obsm["X_rep"] = X.astype(np.float32)

        def fit():
            """Fit a fresh model with the selected seed."""
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
        """Create a small reproducible dataset for clustering regressions."""
        X, _ = make_blobs(n_samples=300, n_features=10, centers=4, cluster_std=3.0, random_state=0)
        adata = ad.AnnData(X=np.zeros((X.shape[0], 2), dtype=np.float32))
        adata.obsm["X_rep"] = X.astype(np.float32)
        return adata

    @staticmethod
    def _model_params():
        """Use deterministic CPU models without progress bars."""
        return {"random_state": 0, "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False}}

    def test_non_consecutive_n_clusters(self):
        """Fit neighbors of every requested K and predict with the selected model."""
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
        """A tuple range and its explicit list produce identical stability."""
        adata = self._blobs_adata()
        autok_tuple = cc.tl.ClusterAutoK(n_clusters=(2, 4), model_params=self._model_params(), max_runs=2)
        autok_list = cc.tl.ClusterAutoK(n_clusters=[2, 3, 4], model_params=self._model_params(), max_runs=2)
        autok_tuple.fit(adata, use_rep="X_rep")
        autok_list.fit(adata, use_rep="X_rep")

        assert autok_tuple.n_clusters == autok_list.n_clusters == [2, 3, 4]
        np.testing.assert_allclose(autok_tuple.stability, autok_list.stability)

    def test_n_clusters_lower_than_two_ignored(self):
        """Ignore unsupported counts and reject an entirely invalid range."""
        with pytest.warns(UserWarning):
            autok = cc.tl.ClusterAutoK(n_clusters=(1, 4))
        assert autok.n_clusters == [2, 3, 4]
        with pytest.raises(ValueError):
            cc.tl.ClusterAutoK(n_clusters=[1])

    def test_load_legacy_model(self):
        """Convert the stored border counts in the legacy multi-run model."""
        # Saved with n_clusters=(2, 5), when `n_clusters` also stored the neighbors 1 and 6
        autok = cc.tl.ClusterAutoK.load("tests/_models/cellcharter_autok_mibitof")
        assert autok.n_clusters == [2, 3, 4, 5]
        assert autok.best_k in autok.n_clusters

    @pytest.mark.parametrize("lower", [2, 3])
    def test_load_legacy_single_run_model(self, tmp_path, lower):
        """Legacy single-run models discard border counts before they are refitted."""
        path = tmp_path / "legacy"
        copytree("tests/_models/cellcharter_autok_mibitof", path)
        with (path / "params.pickle").open("rb") as f:
            params = pickle.load(f)
        params["n_clusters"] = list(range(lower - 1, 7))
        params["max_runs"] = 1
        with (path / "params.pickle").open("wb") as f:
            pickle.dump(params, f)
        with (path / "attributes.pickle").open("rb") as f:
            attributes = pickle.load(f)
        attributes["stability"] = None
        attributes["labels"] = {k: labels[:1] for k, labels in attributes["labels"].items() if k >= lower - 1}
        with (path / "attributes.pickle").open("wb") as f:
            pickle.dump(attributes, f)

        autok = cc.tl.ClusterAutoK.load(path)
        assert autok.n_clusters == list(range(lower, 6))
        assert autok.stability is None
        autok.fit(self._blobs_adata(), use_rep="X_rep")
        assert sorted(autok.labels) == list(range(lower - 1, 7))

    @pytest.mark.parametrize("max_runs", [1, 2])
    @pytest.mark.parametrize("n_clusters", [[2, 3, 4], [2, 5, 9]])
    def test_save_load_preserves_requested_counts(self, tmp_path, max_runs, n_clusters):
        """Loading current models preserves all requested counts and predictions."""
        adata = self._blobs_adata()
        autok = cc.tl.ClusterAutoK(n_clusters=n_clusters, max_runs=max_runs, model_params=self._model_params())
        autok.fit(adata, use_rep="X_rep")
        autok.save(tmp_path)

        loaded = cc.tl.ClusterAutoK.load(tmp_path)
        assert loaded.n_clusters == n_clusters
        assert loaded._fit_n_clusters == autok._fit_n_clusters
        if max_runs == 1:
            assert loaded.stability is None
        else:
            np.testing.assert_array_equal(loaded.stability, autok.stability)
        for k in n_clusters:
            np.testing.assert_array_equal(
                loaded.predict(adata, use_rep="X_rep", k=k), autok.predict(adata, use_rep="X_rep", k=k)
            )
