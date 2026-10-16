from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Union

import numpy as np

AnyRandom = Union[int, np.random.RandomState, None]


def str2list(value: Union[str, list]) -> list:
    """Check whether value is a string. If so, converts into a list containing value."""
    return [value] if isinstance(value, str) else value


# Lightning logs under a different namespace depending on whether it is installed as `pytorch_lightning` or `lightning`.
_LIGHTNING_LOGGERS = ("pytorch_lightning", "lightning.pytorch", "lightning.fabric", "lightning_fabric")


@contextmanager
def silence_lightning(level: int = logging.ERROR) -> Iterator[None]:
    """Temporarily raise the level of the PyTorch Lightning loggers to hide messages such as ``LOCAL_RANK``."""
    loggers = [logging.getLogger(name) for name in _LIGHTNING_LOGGERS]
    previous_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(max(level, lg.getEffectiveLevel()))
    try:
        yield
    finally:
        for lg, previous_level in zip(loggers, previous_levels):
            lg.setLevel(previous_level)
