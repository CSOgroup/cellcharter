import numpy as np
import squidpy as sq

import cellcharter as cc

GROUP_KEY = "batch"
LABEL_KEY = "Cell_class"
key = f"{GROUP_KEY}_{LABEL_KEY}_enrichment"

adata = sq.datasets.merfish()


class TestEnrichment:
    def test_enrichment(self):
        cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)

        assert key in adata.uns
        assert "enrichment" in adata.uns[key]
        assert "params" in adata.uns[key]

        del adata.uns[key]

    def test_copy(self):
        enrichment_dict = cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, copy=True)

        assert "enrichment" in enrichment_dict

        enrichment_dict = cc.gr.enrichment(
            adata, group_key=GROUP_KEY, label_key=LABEL_KEY, copy=True, observed_expected=True
        )

        assert "enrichment" in enrichment_dict
        assert "observed" in enrichment_dict
        assert "expected" in enrichment_dict

    def test_obs_exp(self):
        cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, observed_expected=True)

        assert key in adata.uns
        assert "enrichment" in adata.uns[key]
        assert "observed" in adata.uns[key]
        assert "expected" in adata.uns[key]

        observed = adata.uns[key]["observed"]
        expected = adata.uns[key]["expected"]

        assert observed.shape == (
            adata.obs[GROUP_KEY].cat.categories.shape[0],
            adata.obs[LABEL_KEY].cat.categories.shape[0],
        )
        assert expected.shape[0] == adata.obs[GROUP_KEY].cat.categories.shape[0]
        assert np.all((observed >= 0) & (observed <= 1))
        assert np.all((expected >= 0) & (expected <= 1))
