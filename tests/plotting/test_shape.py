from anndata import AnnData

import cellcharter as cc


class TestPlotBoundaries:
    def test_boundaries(self, codex_adata: AnnData):
        cc.tl.boundaries(codex_adata)
        cc.pl.boundaries(codex_adata, sample="BALBc-1", alpha_boundary=0.5, show_cells=False)

    def test_boundaries_only(self, codex_adata: AnnData):
        cc.tl.boundaries(codex_adata)
        cc.pl.boundaries(codex_adata, sample="BALBc-1", alpha_boundary=0.5)
