from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

DriftFn: Final = Callable[[float, float], float]
DiffusionFn: Final = Callable[[float, float], float]
JumpCoefFn: Final = Callable[[float, float, float], float]


@dataclass(slots=True)
class AdditiveSDE:
    """
    Additive jump-diffusion SDE coefficients.
    dX_t = a(t, X_t) dt + b(t, X_t) dW_t + t^{-sigma} z tilde{N}(dt, dz).
    """

    drift_coefficient: DriftFn
    diffusion_coefficient: DiffusionFn
    sigma: float  # exponent in t^{-sigma} z

    def __post_init__(self):
        if not 0 <= self.sigma < 0.5:
            raise ValueError("sigma must be in [0, 0.5)")

    def jump_coefficient(self, t: float, x: float, jump_size: float) -> float:
        return t ** (-self.sigma) * jump_size
