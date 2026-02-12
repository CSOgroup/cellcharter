from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import anndata as ad

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    doc_header: str
    shape: tuple[int, int]
    url: str
    filename: str


_codex_mouse_spleen = DatasetMetadata(
    name="codex_mouse_spleen",
    doc_header="Pre-processed CODEX dataset of mouse spleen from `Goltsev et al "
    "<https://doi.org/10.1016/j.cell.2018.07.010>`__.",
    shape=(707474, 29),
    url="https://ndownloader.figshare.com/files/38538101",
    filename="codex_mouse_spleen.h5ad",
)


def _default_datasets_dir() -> Path:
    return Path(os.environ.get("CELLCHARTER_DATA_DIR", Path.home() / ".cache" / "cellcharter" / "datasets"))


def _is_hdf5_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as input_file:
        return input_file.read(8) == b"\x89HDF\r\n\x1a\n"


def _normalize_figshare_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == "figshare.com" and parsed.path.startswith("/ndownloader/files/"):
        return f"https://ndownloader.figshare.com{parsed.path}"
    return url


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(f"{destination.suffix}.part")

    request = Request(_normalize_figshare_url(url), headers={"User-Agent": "cellcharter-datasets"})
    progress = None
    try:
        with urlopen(request) as response, tmp_destination.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length and content_length.isdigit() else None
            if tqdm is not None:
                progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {destination.name}")

            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    except (HTTPError, URLError) as error:
        if tmp_destination.exists():
            tmp_destination.unlink()
        raise RuntimeError(f"Failed to download dataset from {url}.") from error
    finally:
        if progress is not None:
            progress.close()

    if not _is_hdf5_file(tmp_destination):
        tmp_destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file from {url} is not a valid HDF5 file. " "Please check network access and dataset URL."
        )

    tmp_destination.replace(destination)


def _resolve_dataset_path(metadata: DatasetMetadata, path: str | Path | None = None) -> Path:
    base_dir = _default_datasets_dir() if path is None else Path(path)
    if base_dir.suffix:
        return base_dir
    return base_dir / metadata.filename


def _fetch_dataset_file(
    metadata: DatasetMetadata, path: str | Path | None = None, force_download: bool = False
) -> Path:
    dataset_path = _resolve_dataset_path(metadata, path=path)
    should_download = force_download or not dataset_path.exists() or not _is_hdf5_file(dataset_path)
    if should_download:
        dataset_path.unlink(missing_ok=True)
        _download_file(metadata.url, dataset_path)
    return dataset_path


def codex_mouse_spleen(path: str | Path | None = None, force_download: bool = False) -> ad.AnnData:
    """Pre-processed CODEX dataset of mouse spleen from `Goltsev et al <https://doi.org/10.1016/j.cell.2018.07.010>`__."""
    dataset_path = _fetch_dataset_file(_codex_mouse_spleen, path=path, force_download=force_download)
    return ad.read_h5ad(dataset_path)


__all__ = ["codex_mouse_spleen"]
