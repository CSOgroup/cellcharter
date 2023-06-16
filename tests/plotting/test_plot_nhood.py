import squidpy as sq
from squidpy._constants._pkg_constants import Key

import cellcharter as cc

_CK = "cell type"
key = Key.uns.nhood_enrichment(_CK)

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
cc.gr.remove_long_links(adata)


class TestPlotNhoodEnrichment:
    def test_annotate(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK)
        cc.pl.nhood_enrichment(adata, cluster_key=_CK, annotate=True)

        del adata.uns[key]

    def test_significance(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, analytical=False, n_perms=100)

        cc.pl.nhood_enrichment(adata, cluster_key=_CK, significance=0.05)
        cc.pl.nhood_enrichment(adata, cluster_key=_CK, annotate=True, significance=0.05)

        del adata.uns[key]
