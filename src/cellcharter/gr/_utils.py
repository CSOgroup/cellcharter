"""Graph utilities."""
from __future__ import annotations

from anndata import AnnData


def _assert_distances_key(adata: AnnData, key: str) -> None:
    if key not in adata.obsp:
        key_added = key.replace("_distances", "")
        raise KeyError(
            f"Spatial distances key `{key}` not found in `adata.obsp`. "
            f"Please run `squidpy.gr.spatial_neighbors(..., key_added={key_added!r})` first."
        )
