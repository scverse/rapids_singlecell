from __future__ import annotations

from enum import Enum


class Empty(Enum):
    token = 0

    def __repr__(self) -> str:
        return "_empty"


_empty = Empty.token
