from __future__ import annotations

from dataclasses import dataclass
from typing import Final


from levy_type.laws.base import JumpLaw
from levy_type.numerics.constants import M_VALUES

__all__: Final = ["AlphaStableAR", "AlphaStableDC"]


def _require_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 2.0:
        raise ValueError("alpha must be in (0, 2)")
    if alpha not in M_VALUES:
        raise ValueError(f"Unsupported alpha={alpha}. Allowed: {sorted(M_VALUES)}")
    return alpha


@dataclass(slots=True)
class AlphaStableAR(JumpLaw):
    alpha: float
    delta: float

    def __post_init__(self) -> None:
        self.alpha = _require_alpha(self.alpha)
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")
        self._M = M_VALUES[self.alpha]

    def inverse_lambda(self, x: float) -> float:
        return x * (self.delta**self.alpha) / self._M

    def inverse_jump_cdf(self, u: float, t: float) -> float:
        return self.delta * (1.0 / (1.0 - u)) ** (1.0 / self.alpha)

    def compensator(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        return 0.0  # symmetry

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        dt = t_curr ** (1 - 2 * sigma) - t_prev ** (1 - 2 * sigma)
        val = self._M * self.delta ** (2 - self.alpha) * self.alpha
        return val * dt / (2 - self.alpha) / (1 - 2 * sigma)


@dataclass(slots=True)
class AlphaStableDC(JumpLaw):
    alpha: float
    h: float
    eps: float

    def __post_init__(self) -> None:
        self.alpha = _require_alpha(self.alpha)
        if self.h <= 0.0:
            raise ValueError("h must be positive")
        if not 0.0 < self.eps < 1.0:
            raise ValueError("eps must be in (0, 1)")
        self._M = M_VALUES[self.alpha]

    def inverse_lambda(self, x: float) -> float:
        return (x * (1.0 - self.eps) * (self.h**self.eps)) ** (1.0 / (1.0 - self.eps))

    def inverse_jump_cdf(self, u: float, t: float) -> float:
        return self._tau(((t * self.h) ** self.eps) / (1.0 - u))

    def compensator(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        return 0.0

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        power = -1 * (2 * self.alpha * sigma + self.alpha * self.eps - self.alpha - 2 * self.eps) / self.alpha
        dt = t_curr**power - t_prev**power
        return (
            self.h ** (self.eps * (2 - self.alpha) / self.alpha)
            * (self._M ** (2 / self.alpha) * self.alpha)
            * dt
            / (2 - self.alpha)
            / power
        )

    def _tau(self, t: float) -> float:
        return (t * self._M) ** (1.0 / self.alpha)
