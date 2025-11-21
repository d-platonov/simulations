from __future__ import annotations

from typing import Final

# Euler-Mascheroni constant.
GAMMA = 0.5772156649015329

# Pivot value E1(1.0) used to choose the inversion regime.
E1_OF_1 = 0.2193839343955206

# Clamp inputs for inv_exp1
MIN_X_INPUT = 1e-300
MAX_X_INPUT = 800.0
MAX_Y_VALUE = 690.1983122333121
MIN_Y_VALUE = 1e-350

# Precomputed M values for efficiency for the Alpha-Stable process simulation.
M_VALUES: Final[dict[float, float]] = {
    0.1: 0.623672,
    0.2: 0.479989,
    0.3: 0.427967,
    0.4: 0.416225,
    0.5: 0.399195,
    0.6: 0.383542,
    0.7: 0.368157,
    0.8: 0.352449,
    0.9: 0.335967,
    1.0: 0.318310,
    1.1: 0.299096,
    1.2: 0.277958,
    1.3: 0.254537,
    1.4: 0.228488,
    1.5: 0.199483,
    1.6: 0.167245,
    1.7: 0.131695,
    1.8: 0.093520,
    1.9: 0.054427,
}
