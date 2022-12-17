import numpy as np
import pytest
import scipy.sparse as sps
import squidpy as sq

import cellcharter as cc


class TestClusterAutoK:
    @pytest.mark.parametrize("func", ["mibitof"])
    def test_spatial_proteomics(self, func: str):
        download_dataset = getattr(sq.datasets, func)
        adata = download_dataset()
        if sps.issparse(adata.X):
            adata.X = adata.X.todense()
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        cc.gr.remove_long_links(adata)
        cc.gr.aggregate_neighbors(adata, n_layers=3)

        model_params = {
            "init_strategy": "sklearn",
            "random_state": 42,
            "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False},
        }
        cls = cc.tl.ClusterAutoK(
            n_clusters=(2, 4), model_class=cc.tl.GaussianMixture, model_params=model_params, max_runs=3
        )
        cls.fit(adata, use_rep="X_cellcharter")

        adata.obs[f"cellcharter_{cls.best_k}"] = cls.predict(adata, use_rep="X_cellcharter", k=cls.best_k)

        assert len(np.unique(adata.obs[f"cellcharter_{cls.best_k}"])) == cls.best_k
