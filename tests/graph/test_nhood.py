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


class TestNhoodEnrichment:
    def test_enrichment(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, only_inter=False, log_fold_change=False)
        enrichment = adata.uns[key]["enrichment"]
        assert np.all((enrichment >= -1) & (enrichment <= 1))

        del adata.uns[key]

    def test_fold_change(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, log_fold_change=True)

        del adata.uns[key]

    def test_nhood_obs_exp(self):
        cc.gr.nhood_enrichment(adata, cluster_key=_CK, only_inter=False, observed_expected=True)
        observed = adata.uns[key]["observed"]
        expected = adata.uns[key]["expected"]

        assert observed.shape[0] == adata.obs[_CK].cat.categories.shape[0]
        assert observed.shape == expected.shape
        assert np.all((observed >= 0) & (observed <= 1))
        assert np.all((expected >= 0) & (expected <= 1))

        del adata.uns[key]

    def test_symmetric(self):
        result = cc.gr.nhood_enrichment(
            adata, cluster_key=_CK, symmetric=True, log_fold_change=True, only_inter=False, copy=True
        )
        assert scipy.linalg.issymmetric(result["enrichment"].values, atol=1e-02)

        result = cc.gr.nhood_enrichment(
            adata, cluster_key=_CK, symmetric=True, log_fold_change=False, only_inter=False, copy=True
        )
        assert scipy.linalg.issymmetric(result["enrichment"].values, atol=1e-02)

        result = cc.gr.nhood_enrichment(
            adata, cluster_key=_CK, symmetric=True, log_fold_change=False, only_inter=True, copy=True
        )
        result["enrichment"][result["enrichment"].isna()] = 0  # issymmetric fails with NaNs
        assert scipy.linalg.issymmetric(result["enrichment"].values, atol=1e-02)

        result = cc.gr.nhood_enrichment(
            adata, cluster_key=_CK, symmetric=True, log_fold_change=True, only_inter=True, copy=True
        )
        result["enrichment"][result["enrichment"].isna()] = 0  # issymmetric fails with NaNs
        assert scipy.linalg.issymmetric(result["enrichment"].values, atol=1e-02)

    def test_perm(self):
        result_analytical = cc.gr.nhood_enrichment(
            adata, cluster_key=_CK, only_inter=True, pvalues=False, observed_expected=True, copy=True
        )
        result_perm = cc.gr.nhood_enrichment(
            adata,
            cluster_key=_CK,
            only_inter=True,
            pvalues=True,
            n_perms=5000,
            observed_expected=True,
            copy=True,
            n_jobs=15,
        )
        np.testing.assert_allclose(result_analytical["enrichment"], result_perm["enrichment"], atol=0.1)
        np.testing.assert_allclose(result_analytical["observed"], result_perm["observed"], atol=0.1)
        np.testing.assert_allclose(result_analytical["expected"], result_perm["expected"], atol=0.1)
