from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import anndata as ad
import numpy as np
import pytest

from . import conftest


def _write_codex(path):
    """Write a small fixture containing the selected and excluded samples."""
    adata = ad.AnnData(np.arange(6).reshape(3, 2))
    adata.obs["sample"] = ["BALBc-1", "MRL-5", "other"]
    adata.write_h5ad(path)


def test_concurrent_codex_loads(tmp_path, monkeypatch):
    """Concurrent workers download once and each read the complete dataset."""
    downloads = []
    barrier = Barrier(4)
    path = tmp_path / "codex.h5ad"

    def download(path):
        downloads.append(path)
        _write_codex(path)

    def load():
        barrier.wait(timeout=10)
        return conftest._load_codex(path)

    monkeypatch.setattr(conftest, "_download_codex", download)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load) for _ in range(4)]
        results = [future.result(timeout=30) for future in futures]

    assert downloads == [path]
    for adata in results:
        assert adata.obs["sample"].tolist() == ["BALBc-1", "MRL-5"]
        np.testing.assert_array_equal(adata.X, np.arange(4).reshape(2, 2))


@pytest.mark.parametrize("error", [OSError, KeyError])
def test_corrupt_codex_cache_is_retried(tmp_path, monkeypatch, error):
    """Unreadable cached HDF5 files are downloaded again before use."""
    path = tmp_path / "codex.h5ad"
    _write_codex(path)
    read_h5ad = ad.read_h5ad
    reads = []
    downloads = []

    def read(path):
        reads.append(path)
        if len(reads) == 1:
            raise error("corrupt HDF5 data")
        return read_h5ad(path)

    def download(path):
        assert not path.exists()
        downloads.append(path)
        _write_codex(path)

    monkeypatch.setattr(conftest.ad, "read_h5ad", read)
    monkeypatch.setattr(conftest, "_download_codex", download)
    monkeypatch.setattr(conftest.time, "sleep", lambda _: None)
    adata = conftest._load_codex(path)

    assert downloads == [path]
    assert len(reads) == 2
    assert adata.obs["sample"].tolist() == ["BALBc-1", "MRL-5"]
