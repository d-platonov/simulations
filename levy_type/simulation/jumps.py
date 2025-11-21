from __future__ import annotations

from enum import IntEnum
from typing import Final

__all__: Final = ["JumpSign"]


class JumpSign(IntEnum):
    """Direction of a jump relative to the origin."""

    NEGATIVE = -1
    POSITIVE = +1
