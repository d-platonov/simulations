from dataclasses import dataclass, field

from levy_type.base_process import BaseProcess
from levy_type.constants import M_VALUES
from levy_type.simulation_config import DCSimulationConfig


@dataclass
class DCProcess(BaseProcess):
    config: DCSimulationConfig
    m: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        alpha = self.config.alpha
        try:
            self.m = M_VALUES[alpha]
        except KeyError as exc:
            raise ValueError(f"Unsupported alpha={alpha}. Allowed: {sorted(M_VALUES)}") from exc

    def tau(self, t: float) -> float:
        return (t * self.m) ** (1 / self.config.alpha)

    def large_jump_cdf_inverse(self, u: float, t: float | None = None) -> float:
        h, eps = self.config.h, self.config.eps
        val = ((t * h) ** eps) / (1 - u)
        return self.tau(val)

    def large_jump_lambda_inverse(self, x: float) -> float:
        h, eps = self.config.h, self.config.eps
        return (x * (1 - eps) * (h**eps)) ** (1 / (1 - eps))

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        # Placeholder (not used for strong error in the current pipeline)
        return 0
