import pytest
import scipy.sparse as sps
import squidpy as sq

import cellcharter as cc


class TestPlotStability:
    @pytest.mark.parametrize("dataset_name", ["mibitof"])
    def test_spatial_proteomics(self, dataset_name: str):
        download_dataset = getattr(sq.datasets, dataset_name)
        adata = download_dataset()
        if sps.issparse(adata.X):
            adata.X = adata.X.todense()
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        cc.gr.remove_long_links(adata)
        cc.gr.aggregate_neighbors(adata, n_layers=3)

        model = cc.tl.ClusterAutoK.load(f"tests/_models/cellcharter_autok_{dataset_name}")

        cc.pl.autok_stability(model)
