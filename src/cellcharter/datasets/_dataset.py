from copy import copy

from squidpy.datasets._utils import AMetadata

_codex_mouse_spleen = AMetadata(
    name="codex_mouse_spleen",
    doc_header="Pre-processed CODEX dataset of mouse spleen from `Goltsev et al "
    "<https://doi.org/10.1016/j.cell.2018.07.010>`__.",
    shape=(707474, 29),
    url="https://figshare.com/ndownloader/files/38538101",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = ["codex_mouse_spleen"]  # noqa: F822
