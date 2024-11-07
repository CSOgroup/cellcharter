import numpy as np
import pytest
from anndata import AnnData

import cellcharter as cc

_CLUSTER_KEY = "cell_type"
_CONDITION_KEY = "sample"
key = f"{_CLUSTER_KEY}_{_CONDITION_KEY}_diff_nhood_enrichment"


class TestDiffNhoodEnrichment:
    def test_enrichment(self, codex_adata: AnnData):
        n_conditions = codex_adata.obs[_CONDITION_KEY].cat.categories.shape[0]
        cc.gr.diff_nhood_enrichment(
            codex_adata, cluster_key=_CLUSTER_KEY, condition_key=_CONDITION_KEY, only_inter=False, log_fold_change=False
        )

        assert len(codex_adata.uns[key]) == n_conditions * (n_conditions - 1) / 2

        for nhood_enrichment in codex_adata.uns[key].values():
            enrichment = nhood_enrichment["enrichment"]
            assert np.all((enrichment >= -1) & (enrichment <= 1))

        del codex_adata.uns[key]

    def test_pvalues(self, codex_adata: AnnData):
        n_conditions = codex_adata.obs[_CONDITION_KEY].cat.categories.shape[0]
        cc.gr.diff_nhood_enrichment(
            codex_adata,
            cluster_key=_CLUSTER_KEY,
            condition_key=_CONDITION_KEY,
            library_key="sample",
            only_inter=True,
            log_fold_change=True,
            pvalues=True,
            n_perms=100,
        )

        assert len(codex_adata.uns[key]) == n_conditions * (n_conditions - 1) / 2

        for nhood_enrichment in codex_adata.uns[key].values():
            assert "pvalue" in nhood_enrichment
            pvalue = nhood_enrichment["pvalue"]
            assert np.all((pvalue >= 0) & (pvalue <= 1))

        del codex_adata.uns[key]

    def test_symmetric_vs_nonsymmetric(self, codex_adata: AnnData):
        # Test symmetric case
        cc.gr.diff_nhood_enrichment(codex_adata, cluster_key=_CLUSTER_KEY, condition_key=_CONDITION_KEY, symmetric=True)
        symmetric_result = codex_adata.uns[key].copy()
        del codex_adata.uns[key]

        # Test non-symmetric case
        cc.gr.diff_nhood_enrichment(
            codex_adata, cluster_key=_CLUSTER_KEY, condition_key=_CONDITION_KEY, symmetric=False
        )
        nonsymmetric_result = codex_adata.uns[key]

        # Results should be different when symmetric=False
        for pair_key in symmetric_result:
            assert not np.allclose(
                symmetric_result[pair_key]["enrichment"], nonsymmetric_result[pair_key]["enrichment"], equal_nan=True
            )

        del codex_adata.uns[key]

    def test_condition_groups(self, codex_adata: AnnData):
        conditions = codex_adata.obs[_CONDITION_KEY].cat.categories[:2]
        cc.gr.diff_nhood_enrichment(
            codex_adata, cluster_key=_CLUSTER_KEY, condition_key=_CONDITION_KEY, condition_groups=conditions
        )

        # Should only have one comparison
        assert len(codex_adata.uns[key]) == 1
        pair_key = f"{conditions[0]}_{conditions[1]}"
        assert pair_key in codex_adata.uns[key]

        del codex_adata.uns[key]

    def test_invalid_inputs(self, codex_adata: AnnData):
        # Test invalid cluster key
        with pytest.raises(KeyError):
            cc.gr.diff_nhood_enrichment(codex_adata, cluster_key="invalid_key", condition_key=_CONDITION_KEY)

        # Test invalid condition key
        with pytest.raises(KeyError):
            cc.gr.diff_nhood_enrichment(codex_adata, cluster_key=_CLUSTER_KEY, condition_key="invalid_key")

        # Test invalid library key when using pvalues
        with pytest.raises(KeyError):
            cc.gr.diff_nhood_enrichment(
                codex_adata,
                cluster_key=_CLUSTER_KEY,
                condition_key=_CONDITION_KEY,
                library_key="invalid_key",
                pvalues=True,
            )

    def test_copy_return(self, codex_adata: AnnData):
        # Test copy=True returns results without modifying adata
        result = cc.gr.diff_nhood_enrichment(
            codex_adata, cluster_key=_CLUSTER_KEY, condition_key=_CONDITION_KEY, copy=True
        )

        assert key not in codex_adata.uns
        assert isinstance(result, dict)
        assert len(result) > 0
