from anndata import AnnData
from shapely import Polygon

import cellcharter as cc


# Test for cc.tl.boundaries, that computes the topological boundaries of sets of cells.
class TestBoundaries:
    def test_boundaries(self, codex_adata: AnnData):
        cc.tl.boundaries(codex_adata)

        boundaries = codex_adata.uns["shape_component"]["boundary"]

        assert isinstance(boundaries, dict)

        # Check if boundaries contains all components of codex_adata
        assert set(boundaries.keys()) == set(codex_adata.obs["component"].cat.categories)

    def test_copy(self, codex_adata: AnnData):
        boundaries = cc.tl.boundaries(codex_adata, copy=True)

        assert isinstance(boundaries, dict)

        # Check if boundaries contains all components of codex_adata
        assert set(boundaries.keys()) == set(codex_adata.obs["component"].cat.categories)


class TestLinearity:
    def test_rectangle(self, codex_adata: AnnData):
        codex_adata.obs["rectangle"] = 1

        polygon = Polygon([(0, 0), (0, 10), (2, 10), (2, 0)])

        codex_adata.uns["shape_rectangle"] = {"boundary": {1: polygon}}
        linearities = cc.tl.linearity(codex_adata, "rectangle", copy=True)
        assert linearities[1] == 1.0

    def test_symmetrical_cross(self, codex_adata: AnnData):
        codex_adata.obs["cross"] = 1

        # Symmetrical cross with arm width of 2 and length of 5
        polygon = Polygon(
            [(0, 5), (0, 7), (5, 7), (5, 12), (7, 12), (7, 7), (12, 7), (12, 5), (7, 5), (7, 0), (5, 0), (5, 5)]
        )

        codex_adata.uns["shape_cross"] = {"boundary": {1: polygon}}
        linearities = cc.tl.linearity(codex_adata, "cross", copy=True)

        # The cross is symmetrical, so the linearity should be 0.5
        assert abs(linearities[1] - 0.5) < 0.01

    def test_thickness(self, codex_adata: AnnData):
        # The thickness of the cross should not influence the linearity
        codex_adata.obs["cross"] = 1

        # Symmetrical cross with arm width of 2 and length of 5
        polygon1 = Polygon(
            [(0, 5), (0, 6), (5, 6), (5, 11), (6, 11), (6, 6), (11, 6), (11, 5), (6, 5), (6, 0), (5, 0), (5, 5)]
        )

        # Symmetrical cross with arm width of 2 and length of 5
        polygon2 = Polygon(
            [(0, 5), (0, 7), (5, 7), (5, 12), (7, 12), (7, 7), (12, 7), (12, 5), (7, 5), (7, 0), (5, 0), (5, 5)]
        )

        codex_adata.uns["shape_cross"] = {"boundary": {1: polygon1}}
        linearities1 = cc.tl.linearity(codex_adata, "cross", copy=True)

        codex_adata.uns["shape_cross"] = {"boundary": {1: polygon2}}
        linearities2 = cc.tl.linearity(codex_adata, "cross", copy=True)

        assert abs(linearities1[1] - linearities2[1]) < 0.01
