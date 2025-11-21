from __future__ import annotations

from scipy.optimize import brentq
from scipy.special import exp1

from levy_type.numerics import MIN_X_INPUT, MAX_X_INPUT, MIN_Y_VALUE, MAX_Y_VALUE


def inv_exp1(y: float) -> float:
    """Compute the inverse of exp1 function using Brent's method."""
    if y < MIN_Y_VALUE:
        return MAX_X_INPUT
    if y > MAX_Y_VALUE:
        return MIN_X_INPUT

    return brentq(lambda x: exp1(x) - y, MIN_X_INPUT, MAX_X_INPUT)
