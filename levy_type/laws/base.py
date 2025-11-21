from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np

from levy_type.simulation.jumps import JumpSign

__all__: Final = ["JumpLaw"]


class JumpLaw(ABC):
    """Interface for Levy jump laws."""

    def sample_jump_times(self, rng: np.random.Generator, t: float) -> list[float]:
        """Generate jump times up to time t using inverse intensity sampling."""
        times: list[float] = []
        exp_sum = rng.exponential()
        inv = self.inverse_lambda
        while (jump_time := inv(exp_sum)) < t:
            times.append(jump_time)
            exp_sum += rng.exponential()
        return times

    def sample_jump_size(self, rng: np.random.Generator, t: float, sign: JumpSign) -> float:
        """Generate a signed jump size at time t using inverse CDF."""
        u = rng.uniform()
        size = self.inverse_jump_cdf(u, t)
        return float(sign) * size

    @abstractmethod
    def inverse_lambda(self, x: float) -> float:
        """Invert cumulative intensity."""

    @abstractmethod
    def inverse_jump_cdf(self, u: float, t: float) -> float:
        """Invert jump size CDF (one-sided)."""

    @abstractmethod
    def compensator(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        """Large-jump compensator over [t_prev, t_curr]."""

    @abstractmethod
    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float, sigma: float) -> float:
        """Variance used for Gaussian small-jump approximation over [t_prev, t_curr]."""
