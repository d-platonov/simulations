from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationParams:
    """Global time/discretization settings for a simulation run."""

    T: float
    N: int
    x0: float
    random_seed: int
