from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__: Final = ["FloatArray", "Path"]


@dataclass(slots=True)
class Path:
    """Time grid and associated process values for a simulated path."""

    times: FloatArray
    values: FloatArray
