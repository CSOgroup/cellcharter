import pytest
import squidpy as sq

try:
    from matplotlib.colormaps import get_cmap
except ImportError:
    from matplotlib.pyplot import get_cmap

import cellcharter as cc

GROUP_KEY = "batch"
LABEL_KEY = "Cell_class"
key = f"{GROUP_KEY}_{LABEL_KEY}_enrichment"

adata = sq.datasets.merfish()
adata_empirical = adata.copy()
cc.gr.enrichment(adata_empirical, group_key=GROUP_KEY, label_key=LABEL_KEY, pvalues=True, n_perms=1000)

adata_analytical = adata.copy()
cc.gr.enrichment(adata_analytical, group_key=GROUP_KEY, label_key=LABEL_KEY)


class TestProportion:
    def test_proportion(self):
        cc.pl.proportion(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)

    def test_groups_labels(self):
        cc.pl.proportion(
            adata,
            group_key=GROUP_KEY,
            label_key=LABEL_KEY,
            groups=adata.obs[GROUP_KEY].cat.categories[:3],
            labels=adata.obs[LABEL_KEY].cat.categories[:4],
        )


class TestPlotEnrichment:
    @pytest.mark.parametrize("adata_enrichment", [adata_analytical, adata_empirical])
    def test_enrichment(self, adata_enrichment):
        cc.pl.enrichment(adata_enrichment, group_key=GROUP_KEY, label_key=LABEL_KEY)

    @pytest.mark.parametrize("adata_enrichment", [adata_analytical, adata_empirical])
    @pytest.mark.parametrize("label_cluster", [False, True])
    @pytest.mark.parametrize("groups", [None, adata.obs[GROUP_KEY].cat.categories[:3]])
    @pytest.mark.parametrize("labels", [None, adata.obs[LABEL_KEY].cat.categories[:4]])
    @pytest.mark.parametrize("size_threshold", [1, 2.5])
    @pytest.mark.parametrize("palette", [None, "coolwarm", get_cmap("coolwarm")])
    @pytest.mark.parametrize("figsize", [None, (10, 8)])
    @pytest.mark.parametrize("alpha,edgecolor", [(1, "red"), (0.5, "blue")])
    def test_params(
        self, adata_enrichment, label_cluster, groups, labels, size_threshold, palette, figsize, alpha, edgecolor
    ):
        cc.pl.enrichment(
            adata_enrichment,
            group_key=GROUP_KEY,
            label_key=LABEL_KEY,
            label_cluster=label_cluster,
            groups=groups,
            labels=labels,
            size_threshold=size_threshold,
            palette=palette,
            figsize=figsize,
            alpha=alpha,
            edgecolor=edgecolor,
        )

    def test_no_pvalues(self):
        # If the enrichment data is not present, it should raise an error
        with pytest.raises(ValueError):
            cc.pl.enrichment(adata_analytical, "group_key", "label_key", show_pvalues=True)

        with pytest.raises(ValueError):
            cc.pl.enrichment(adata_analytical, "group_key", "label_key", significance=0.01)

        with pytest.raises(ValueError):
            cc.pl.enrichment(adata_analytical, "group_key", "label_key", significant_only=True)

    def test_obs_exp(self):
        cc.gr.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY, observed_expected=True)
        cc.pl.enrichment(adata, group_key=GROUP_KEY, label_key=LABEL_KEY)

    def test_enrichment_no_enrichment_data(self):
        with pytest.raises(ValueError):
            cc.pl.enrichment(adata, "group_key", "label_key")

    def test_size_threshold_zero(self):
        with pytest.raises(ValueError):
            cc.pl.enrichment(adata_empirical, group_key=GROUP_KEY, label_key=LABEL_KEY, size_threshold=0)
