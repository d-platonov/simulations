from dataclasses import dataclass, field

from levy_type.base_process import BaseProcess
from levy_type.constants import M_VALUES
from levy_type.simulation_config import ARSimulationConfig


@dataclass
class ARProcess(BaseProcess):
    config: ARSimulationConfig
    m: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        alpha = self.config.alpha
        try:
            self.m = M_VALUES[alpha]
        except KeyError as exc:
            raise ValueError(f"Unsupported alpha={alpha}. Allowed: {sorted(M_VALUES)}") from exc

    def large_jump_cdf_inverse(self, u: float, t: float | None = None) -> float:
        eps, alpha = self.config.delta, self.config.alpha
        return (1 / (1 - u)) ** (1 / alpha) * eps

    def large_jump_lambda_inverse(self, x: float) -> float:
        eps, alpha = self.config.delta, self.config.alpha
        return x * (eps**alpha) / self.m

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        # Placeholder (not used for strong error in the current pipeline)
        return 0
