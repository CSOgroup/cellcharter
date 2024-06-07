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


@pytest.fixture()
def codex_adata() -> ad.AnnData:
    adata = ad.read_h5ad("tests/_data/codex_adata.h5ad")
    return adata
