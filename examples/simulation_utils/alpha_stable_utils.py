import numpy as np
from scipy.special import gamma

from levy_type.numerics import M_VALUES


def expected_number_of_jumps(t: float, alpha: float, delta: float) -> float:
    return t * (2 * M_VALUES[alpha]) / (delta**alpha)


def get_delta_ar(n_jumps: float, alpha: float, t: float = 1.0):
    m = M_VALUES[alpha]
    return (2 * m * t / n_jumps) ** (1 / alpha)


def get_eps_dc(alpha: float, sigma: float):
    return alpha * sigma if sigma > 0.0 else 0.1


def get_h_dc(n_jumps: float, eps_dc: float, t: float = 1.0):
    return ((2 * t ** (1 - eps_dc)) / (n_jumps * (1 - eps_dc))) ** (1 / eps_dc)


def true_alpha_stable_moment(p: float, alpha: float, t: float | np.ndarray, sigma: float = 0.0) -> float | np.ndarray:
    """Calculate true p-th moment of a symmetric alpha-stable process (sigma = 0)."""
    power = 1 - sigma * alpha
    time_scale = (t**power / power) ** (p / alpha)
    return time_scale * (2**p * gamma((1 + p) / 2) * gamma(1 - p / alpha)) / (gamma(1 - p / 2) * gamma(1 / 2))
