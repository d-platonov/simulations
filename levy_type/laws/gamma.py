from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import exp1

from levy_type.laws.base import JumpLaw
from levy_type.numerics.special_functions import inv_exp1

__all__: Final = ["GammaAR", "GammaDC"]


@dataclass(slots=True)
class GammaAR(JumpLaw):
    gamma: float
    lam: float
    delta: float

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")

    def inverse_lambda(self, x: float) -> float:
        return x / (self.gamma * exp1(self.lam * self.delta))

    def inverse_jump_cdf(self, u: float, t: float) -> float:
        y = (1.0 - u) * exp1(self.lam * self.delta)
        return inv_exp1(y) / self.lam

    def compensator(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        dt = t_curr ** (1 - sigma) - t_prev ** (1 - sigma)
        return (self.gamma / self.lam) * np.exp(-self.lam * self.delta) * dt / (1 - sigma)

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        dt = t_curr ** (1 - 2 * sigma) - t_prev ** (1 - 2 * sigma)
        return (
            (self.gamma / self.lam**2)
            * (1 - np.exp(-self.lam * self.delta) * (1 + self.lam * self.delta))
            * dt
            / (1 - 2 * sigma)
        )


@dataclass(slots=True)
class GammaDC(JumpLaw):
    gamma: float
    lam: float
    h: float
    eps: float

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive")
        if self.h <= 0.0:
            raise ValueError("h must be positive")
        if not 0.0 < self.eps < 1.0:
            raise ValueError("eps must be in (0, 1)")

    def inverse_lambda(self, x: float) -> float:
        return (x * (1.0 - self.eps) * (self.h**self.eps)) ** (1.0 / (1.0 - self.eps))

    def inverse_jump_cdf(self, u: float, t: float) -> float:
        return self._tau(((t * self.h) ** self.eps) / (1.0 - u))

    def compensator(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        t_mid = 0.5 * (t_prev + t_curr)
        tau_mid = self._tau((t_mid * self.h) ** self.eps)
        return (self.gamma / self.lam) * (t_mid ** (-sigma)) * np.exp(-self.lam * tau_mid) * (t_curr - t_prev)

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        t_mid = 0.5 * (t_prev + t_curr)
        tau_mid = self._tau((t_mid * self.h) ** self.eps)
        return (
            (self.gamma / self.lam**2)
            * (t_mid ** (-2 * sigma))
            * (1 - np.exp(-self.lam * tau_mid) * (1 + self.lam * tau_mid))
            * (t_curr - t_prev)
        )

    def _tau(self, t: float) -> float:
        value = inv_exp1(1.0 / (self.gamma * t)) / self.lam
        return float(np.real(value))
