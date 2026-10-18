from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Union

import numpy as np

AnyRandom = Union[int, np.random.RandomState, None]


def str2list(value: str | list) -> list:
    """Check whether value is a string. If so, converts into a list containing value."""
    return [value] if isinstance(value, str) else value


# Lightning logs under a different namespace depending on whether it is installed as `pytorch_lightning` or `lightning`.
_LIGHTNING_LOGGERS = ("pytorch_lightning", "lightning.pytorch", "lightning.fabric", "lightning_fabric")


_silence_lock = threading.Lock()
_silence_depth = 0
_silence_previous_levels: list[int] = []


@contextmanager
def silence_lightning() -> Iterator[None]:
    """Temporarily raise the level of the PyTorch Lightning loggers to hide messages such as ``LOCAL_RANK``.

    Nested and concurrent (e.g. multi-threaded) uses are reference-counted: the first one to enter changes the levels
    and the last one to exit restores them.
    """
    global _silence_depth, _silence_previous_levels
    loggers = [logging.getLogger(name) for name in _LIGHTNING_LOGGERS]
    with _silence_lock:
        if _silence_depth == 0:
            _silence_previous_levels = [lg.level for lg in loggers]
            for lg in loggers:
                lg.setLevel(max(logging.ERROR, lg.getEffectiveLevel()))
        _silence_depth += 1
    try:
        yield
    finally:
        with _silence_lock:
            _silence_depth -= 1
            if _silence_depth == 0:
                for lg, previous_level in zip(loggers, _silence_previous_levels):
                    lg.setLevel(previous_level)
