import numpy as np
from anndata import AnnData

import cellcharter as cc

_CLUSTER_KEY = "cell_type"
_CONDITION_KEY = "sample"
key = f"{_CLUSTER_KEY}_{_CONDITION_KEY}_diff_nhood_enrichment"


class TestDiffNhoodEnrichment:
    def test_enrichment(self, codex_adata: AnnData):
        print(codex_adata)
        n_conditions = codex_adata.obs[_CONDITION_KEY].cat.categories.shape[0]
        cc.gr.diff_nhood_enrichment(
            codex_adata, cluster_key=_CLUSTER_KEY, condition_key=_CONDITION_KEY, only_inter=False, log_fold_change=False
        )

        assert len(codex_adata.uns[key]) == n_conditions * (n_conditions - 1) / 2

        for nhood_enrichment in codex_adata.uns[key].values():
            enrichment = nhood_enrichment["enrichment"]
            assert np.all((enrichment >= -1) & (enrichment <= 1))

        del codex_adata.uns[key]
