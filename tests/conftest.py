import time
from urllib.error import HTTPError

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from squidpy._constants._pkg_constants import Key

_adata = sc.read("tests/_data/test_data.h5ad")
_adata.raw = _adata.copy()


@pytest.fixture()
def non_visium_adata() -> ad.AnnData:
    non_visium_coords = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])
    adata = ad.AnnData(X=non_visium_coords, dtype=int)
    adata.obsm[Key.obsm.spatial] = non_visium_coords
    return adata


@pytest.fixture()
def adata() -> ad.AnnData:
    return _adata.copy()


@pytest.fixture(scope="session")
def codex_adata() -> ad.AnnData:
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            adata = sc.read(
                "tests/_data/codex_adata.h5ad", backup_url="https://figshare.com/ndownloader/files/46832722"
            )
            return adata[adata.obs["sample"].isin(["BALBc-1", "MRL-5"])].copy()
        except HTTPError as e:
            if attempt == max_retries - 1:  # Last attempt
                pytest.skip(f"Failed to download test data after {max_retries} attempts: {str(e)}")
            time.sleep(retry_delay)
