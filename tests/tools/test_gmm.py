import pytest
import scipy.sparse as sps
import squidpy as sq

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
