import numpy as np
import pandas as pd
import scipy.sparse as sps
from anndata import AnnData
from squidpy._constants._pkg_constants import Key

import cellcharter as cc


def test_nhood_enrichment_only_inter_sets_diagonal_to_nan():
    adata = AnnData(X=np.ones((4, 2)))
    adata.obs["cluster"] = pd.Categorical(["0", "0", "1", "1"])
    adata.obsp[Key.obsp.spatial_conn()] = sps.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ]
        )
    )

    result = cc.gr.nhood_enrichment(adata, cluster_key="cluster", only_inter=True, observed_expected=True, copy=True)

    assert np.all(np.isnan(np.diag(result["observed"])))
    assert np.all(np.isnan(np.diag(result["expected"])))
    assert np.all(np.isnan(np.diag(result["enrichment"])))
