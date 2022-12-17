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


class TestPlotNhoodEnrichment:
    def test_standard(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, only_inter=False)
        enrichment = cc.pl.nhood_enrichment(adata, cluster_key=_CK, return_enrichment=True)

        assert np.all((enrichment >= -1) & (enrichment <= 1))

        del adata.uns[key]

    def test_fold_change(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK)
        cc.pl.nhood_enrichment(adata, cluster_key=_CK, fold_change=True)

        del adata.uns[key]

    def test_symmetric(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, symmetric=True)
        enrichment = cc.pl.nhood_enrichment(adata, cluster_key=_CK, return_enrichment=True)
        assert scipy.linalg.issymmetric(enrichment.values, rtol=1e-05, atol=1e-08)

        enrichment = cc.pl.nhood_enrichment(adata, cluster_key=_CK, fold_change=True, return_enrichment=True)
        assert scipy.linalg.issymmetric(enrichment.values, rtol=1e-05, atol=1e-08)

        del adata.uns[key]

    def test_annotate(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK)
        cc.pl.nhood_enrichment(adata, cluster_key=_CK, annotate=True)

        del adata.uns[key]
