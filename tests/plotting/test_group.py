import squidpy as sq

import cellcharter as cc

GROUP_KEY = "batch"
LABEL_KEY = "Cell_class"
key = f"{GROUP_KEY}_{LABEL_KEY}_enrichment"

adata = sq.datasets.merfish()


class TestPlotEnrichment:
    def test_enrichment(self):
        cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)
        cc.pl.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)

        del adata.uns[key]

    def test_params(self):
        cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)
        cc.pl.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, cluster_labels=False)
        cc.pl.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, size_threshold=100)
        cc.pl.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, color_threshold=0)
        cc.pl.enrichment(
            adata,
            group_key=GROUP_KEY,
            label_key=LABEL_KEY,
            groups=adata.obs[GROUP_KEY].cat.categories[:3],
            labels=adata.obs[LABEL_KEY].cat.categories[:4],
        )

        del adata.uns[key]

    def test_obs_exp(self):
        cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, observed_expected=True)
        cc.pl.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)
