from anndata import AnnData

import cellcharter as cc


# Test for cc.tl.boundaries, that computes the topological boundaries of sets of cells.
class TestBoundaries:
    def test_boundaries(self, codex_adata: AnnData):
        cc.tl.boundaries(codex_adata)

        boundaries = codex_adata.uns["boundaries_component"]

        assert isinstance(boundaries, dict)

        # Check if boundaries contains all components of codex_adata
        assert set(boundaries.keys()) == set(codex_adata.obs["component"].cat.categories)

    def test_copy(self, codex_adata: AnnData):
        boundaries = cc.tl.boundaries(codex_adata, copy=True)

        assert isinstance(boundaries, dict)

        # Check if boundaries contains all components of codex_adata
        assert set(boundaries.keys()) == set(codex_adata.obs["component"].cat.categories)
