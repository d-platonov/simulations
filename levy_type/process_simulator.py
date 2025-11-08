from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from levy_type.base_process import BaseProcess, JumpSign
from levy_type.process_factory import ProcessFactory
from levy_type.simulation_config import SimulationConfig


@dataclass(slots=True)
class Path:
    times: np.ndarray
    values: np.ndarray


class ProcessSimulator:
    """
    Simulator for a jump-driven SDE.
    """

    def __init__(self, config: SimulationConfig, approximate_small_jumps: bool = True) -> None:
        self.config = config
        self.process = ProcessFactory.create_process(config)
        self.approximate_small_jumps = approximate_small_jumps

    def simulate(self) -> Path:
        """Simulate a single path of the process."""
        jump_times, jump_sizes = self._generate_all_jumps(self.process, float(self.config.T))

        # Create time grid combining uniform grid and jump times
        time_grid = self._create_time_grid(jump_times)

        # Map jumps to grid indices
        jump_map = self._create_jump_map(time_grid, jump_times, jump_sizes)

        values = self._construct_path(time_grid, jump_map)

        return Path(times=time_grid, values=values)

    def simulate_many(self, n: int) -> list[Path]:
        """Simulate n independent paths."""
        return [self.simulate() for _ in range(n)]

    def _generate_all_jumps(self, process: BaseProcess, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate all large jumps (positive and negative) up to time t."""
        positive_jumps = self._generate_one_sided_jumps(process, t, JumpSign.POSITIVE)
        negative_jumps = self._generate_one_sided_jumps(process, t, JumpSign.NEGATIVE)
        return self._merge_jumps(positive_jumps, negative_jumps)

    @staticmethod
    def _generate_one_sided_jumps(process: BaseProcess, t: float, sign: JumpSign) -> tuple[np.ndarray, np.ndarray]:
        """Generate jumps of a single sign (positive or negative)."""
        times = np.asarray(process.sample_large_jump_times(t), dtype=float)

        if times.size == 0:
            return times, np.empty(0, dtype=float)

        sizes = np.array([process.sample_large_jump_sizes(tt, sign) for tt in times], dtype=float)

        return times, sizes

    @staticmethod
    def _merge_jumps(
        positive_jumps: tuple[np.ndarray, np.ndarray],
        negative_jumps: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge positive and negative jumps without sorting."""
        t_pos, z_pos = positive_jumps
        t_neg, z_neg = negative_jumps

        if t_pos.size == 0 and t_neg.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        all_times = np.concatenate([t_pos, t_neg])
        all_sizes = np.concatenate([z_pos, z_neg])

        return all_times, all_sizes

    def _create_time_grid(self, jump_times: np.ndarray) -> np.ndarray:
        """Create time grid combining uniform grid and jump times."""
        regular_times = np.linspace(0.0, self.config.T, self.config.N)
        return np.union1d(regular_times, jump_times)

    @staticmethod
    def _create_jump_map(
        time_grid: np.ndarray,
        jump_times: np.ndarray,
        jump_sizes: np.ndarray,
    ) -> np.ndarray:
        """Map jumps to time grid indices."""
        jump_map = np.zeros_like(time_grid, dtype=float)

        if jump_times.size > 0:
            indices = np.searchsorted(time_grid, jump_times)
            np.add.at(jump_map, indices, jump_sizes)

        return jump_map

    def _construct_path(
        self,
        time_grid: np.ndarray,
        jump_map: np.ndarray,
    ) -> np.ndarray:
        """Construct the process path along the time grid."""
        n_times = len(time_grid)
        values = self._initialize_path(time_grid)

        current_value = values[0]
        for i in range(1, n_times):
            current_value = self._compute_next_step(i, time_grid, values, jump_map)
            values[i] = current_value

        # Explicitly set final value at T
        values[-1] = current_value

        return values

    def _initialize_path(self, time_grid: np.ndarray) -> np.ndarray:
        """Initialize path values array with starting value."""
        values = np.empty_like(time_grid, dtype=float)
        values[0] = float(self.config.x_0)
        return values

    def _compute_next_step(
        self,
        i: int,
        time_grid: np.ndarray,
        values: np.ndarray,
        jump_map: np.ndarray,
    ) -> float:
        """Compute the process value at the next time step."""
        t_prev = float(time_grid[i - 1])
        t_curr = float(time_grid[i])
        x_prev = float(values[i - 1])
        dt = t_curr - t_prev

        x_next = x_prev

        x_next += self._apply_drift(t_prev, x_prev, dt)
        x_next += self._apply_diffusion(t_prev, x_prev, dt)

        jump_size = float(jump_map[i])
        if jump_size != 0.0:
            x_next += self._apply_large_jump(t_curr, x_prev, jump_size)

        if self.approximate_small_jumps:
            x_next += self._apply_small_jumps(x_prev, t_prev, t_curr, dt)

        return x_next

    def _apply_drift(self, t: float, x: float, dt: float) -> float:
        """Apply drift increment."""
        return self.process.drift_coefficient(t, x) * dt

    def _apply_diffusion(self, t: float, x: float, dt: float) -> float:
        """Apply diffusion increment."""
        sigma = self.process.diffusion_coefficient(t, x)

        if sigma == 0.0:
            return 0.0

        return sigma * np.sqrt(dt) * self.process.rng.standard_normal()

    def _apply_large_jump(self, t: float, x: float, jump_size: float) -> float:
        """Apply large jump increment."""
        return self.process.jump_coefficient(t, x, jump_size)

    def _apply_small_jumps(
        self,
        x_prev: float,
        t_prev: float,
        t_curr: float,
        dt: float,
    ) -> float:
        """Apply small jump Gaussian approximation."""
        if dt <= 0.0:
            return 0.0

        variance = self.process.small_jump_variance(x_prev, t_prev, t_curr)

        if variance <= 0.0:
            return 0.0

        return np.sqrt(variance) * self.process.rng.standard_normal()
