import numpy as np
import scipy
import squidpy as sq
from squidpy._constants._pkg_constants import Key

import cellcharter as cc

_CK = "cell type"
key = Key.uns.nhood_enrichment(_CK)

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
cc.gr.remove_long_links(adata)


class TestNhoodEnrichment:
    def test_plot_nhood_enrichment(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, only_inter=False)
        observed = adata.uns[key]["observed"]
        expected = adata.uns[key]["expected"]

        assert observed.shape[0] == adata.obs[_CK].cat.categories.shape[0]
        assert expected.shape[0] == adata.obs[_CK].cat.categories.shape[0]
        assert observed.shape == expected.shape
        assert np.all((observed >= 0) & (observed <= 1))
        assert np.all((expected >= 0) & (expected <= 1))

        del adata.uns[key]

    def test_only_inter(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK)
        assert np.all(np.diag(np.isnan(adata.uns[key]["expected"])))

    def test_symmetric(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, symmetric=True)

        assert scipy.linalg.issymmetric(adata.uns[key]["observed"].values, rtol=1e-05, atol=1e-08)
        assert scipy.linalg.issymmetric(adata.uns[key]["expected"].values, rtol=1e-05, atol=1e-08)
