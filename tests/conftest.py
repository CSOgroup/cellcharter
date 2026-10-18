import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from filelock import FileLock
from squidpy._constants._pkg_constants import Key

_adata = sc.read("tests/_data/test_data.h5ad")
_adata.raw = _adata.copy()
_CODEX_PATH = Path("tests/_data/codex_adata.h5ad")
_CODEX_URL = "https://ndownloader.figshare.com/files/46832722"


def _is_hdf5_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as input_file:
        return input_file.read(8) == b"\x89HDF\r\n\x1a\n"


def _download_codex(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.part")
    request = Request(_CODEX_URL, headers={"User-Agent": "cellcharter-tests"})

    with urlopen(request, timeout=60) as response, tmp_path.open("wb") as output:
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)

    if not _is_hdf5_file(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise OSError("Downloaded codex fixture is not a valid HDF5 file.")

    tmp_path.replace(path)


@pytest.fixture()
def non_visium_adata() -> ad.AnnData:
    non_visium_coords = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])
    adata = ad.AnnData(X=non_visium_coords.astype(int))
    adata.obsm[Key.obsm.spatial] = non_visium_coords
    return adata


@pytest.fixture()
def adata() -> ad.AnnData:
    return _adata.copy()


@pytest.fixture(scope="session")
def _codex_adata() -> ad.AnnData:
    """Load the two CODEX samples once per test session."""
    return _load_codex(_CODEX_PATH)


@pytest.fixture()
def codex_adata(_codex_adata: ad.AnnData) -> ad.AnnData:
    """Give each test its own copy of the CODEX samples."""
    return _codex_adata.copy()


def _load_codex(path: Path) -> ad.AnnData:
    """Download and read the shared fixture without races between pytest workers."""
    max_retries = 3
    retry_delay = 5  # seconds

    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        for attempt in range(max_retries):
            try:
                if not _is_hdf5_file(path):
                    path.unlink(missing_ok=True)
                    _download_codex(path)
                adata = ad.read_h5ad(path)
            except (HTTPError, URLError, OSError, KeyError, ValueError) as e:
                path.unlink(missing_ok=True)  # Force re-download on next attempt
                if attempt == max_retries - 1:  # Last attempt
                    pytest.skip(f"Failed to download test data after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay)
            else:
                break

    adata.obs_names_make_unique()
    return adata[adata.obs["sample"].isin(["BALBc-1", "MRL-5"])].copy()
