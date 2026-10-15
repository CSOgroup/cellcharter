import random

import numpy as np
import pytest
import scipy.sparse as sps
import squidpy as sq
import torch
from sklearn.datasets import make_blobs

import cellcharter as cc


class TestCluster:
    @pytest.mark.parametrize("dataset_name", ["mibitof"])
    def test_sparse(self, dataset_name: str):
        download_dataset = getattr(sq.datasets, dataset_name)
        adata = download_dataset()
        adata.X = sps.csr_matrix(adata.X)

        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        cc.gr.remove_long_links(adata)

        gmm = cc.tl.Cluster(n_clusters=(10))

        # Check if fit raises a ValueError
        with pytest.raises(ValueError):
            gmm.fit(adata, use_rep=None)


class TestGaussianMixture:
    TRAINER_PARAMS = {"accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False}

    @staticmethod
    def _data():
        X, _ = make_blobs(n_samples=1500, n_features=10, centers=8, cluster_std=3.0, random_state=0)
        return X.astype(np.float32)

    def _fit_labels(self, X, random_state):
        gmm = cc.tl.GaussianMixture(n_clusters=8, random_state=random_state, trainer_params=dict(self.TRAINER_PARAMS))
        gmm.fit(X)
        return np.asarray(gmm.predict(X))

    def test_random_state_reproducible(self):
        X = self._data()
        np.testing.assert_array_equal(self._fit_labels(X, 0), self._fit_labels(X, 0))

    def test_random_state_different_seeds(self):
        X = self._data()
        assert not np.array_equal(self._fit_labels(X, 0), self._fit_labels(X, 1))

    def test_random_state_preserves_global_rng(self):
        X = self._data()

        def draw():
            return random.random(), np.random.rand(), torch.rand(1).item()

        random.seed(123), np.random.seed(123), torch.manual_seed(123)
        expected = draw()
        random.seed(123), np.random.seed(123), torch.manual_seed(123)
        self._fit_labels(X, 0)
        assert draw() == expected
