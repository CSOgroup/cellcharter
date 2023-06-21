import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from squidpy._constants._pkg_constants import Key

_adata = sc.read("tests/_data/test_data.h5ad")
_adata.raw = _adata.copy()


@pytest.fixture()
def non_visium_adata() -> AnnData:
    non_visium_coords = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])
    adata = AnnData(X=non_visium_coords, dtype=int)
    adata.obsm[Key.obsm.spatial] = non_visium_coords
    return adata


@pytest.fixture()
def adata() -> AnnData:
    return _adata.copy()


@pytest.fixture()
def codex_adata() -> AnnData:
    adata = sc.read("tests/_data/codex_adata.h5ad")
    return adata
