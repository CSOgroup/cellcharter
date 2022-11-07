from __future__ import annotations

from typing import Union


def str2list(value: Union[str, list]) -> list:
    """Check whether value is a string. If so, converts into a list containing value."""
    return [value] if isinstance(value, str) else value
