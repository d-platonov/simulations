from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from levy_type.constants import DEFAULT_SIGMA
from levy_type.simulation_config import SimulationConfig


class JumpSign(IntEnum):
    NEGATIVE = -1
    POSITIVE = +1


@dataclass
class BaseProcess(ABC):
    config: SimulationConfig
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.random_seed)

    @staticmethod
    def drift_coefficient(t: float, x: float) -> float:
        return 0

    @staticmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        return 0

    @staticmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        if not (-1 < z < 1):
            return 0
        return z * np.cos(x) * (t ** (-DEFAULT_SIGMA))

    @abstractmethod
    def large_jump_lambda_inverse(self, x: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def large_jump_cdf_inverse(self, u: float, t: float | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        raise NotImplementedError

    def sample_large_jump_times(self, t: float) -> list[float]:
        times = []
        exp_sum = self.rng.exponential()
        while (jump_time := self.large_jump_lambda_inverse(exp_sum)) < t:
            times.append(jump_time)
            exp_sum += self.rng.exponential()
        return times

    def sample_large_jump_sizes(self, t: float, sign: JumpSign) -> float:
        u = self.rng.uniform()
        size = self.large_jump_cdf_inverse(u, t)
        return float(sign) * size
