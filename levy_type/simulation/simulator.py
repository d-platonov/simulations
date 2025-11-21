from __future__ import annotations

from typing import Final, Mapping

import numpy as np

from levy_type.laws.base import JumpLaw
from levy_type.sde import AdditiveSDE
from levy_type.simulation.factory import JumpContext, build_jump_contexts
from levy_type.simulation.jumps import JumpSign
from levy_type.simulation.params import SimulationParams
from levy_type.simulation.path import FloatArray, Path

__all__: Final = ["ProcessSimulator"]


class ProcessSimulator:
    """Simulator for a jump-driven SDE.

    Pass a single `JumpLaw` for symmetric processes, or a {JumpSign: JumpLaw}
    mapping to use different laws/parameters for positive vs negative jumps.
    """

    def __init__(
        self,
        sde: AdditiveSDE,
        jump_law: Mapping[JumpSign, JumpLaw],
        params: SimulationParams,
        approximate_small_jumps: bool = False,
        compensate_large_jumps: bool = False,
    ) -> None:
        self.sde = sde
        self.params = params
        self.approximate_small_jumps = approximate_small_jumps
        self.compensate_large_jumps = compensate_large_jumps

        self.jump_contexts: tuple[JumpContext, ...] = build_jump_contexts(jump_law, base_seed=params.random_seed)
        # Independent RNG streams for diffusion and Gaussian small-jump approximation
        offset = len(self.jump_contexts)
        self._diffusion_rng = np.random.default_rng(params.random_seed + offset)
        self._small_jump_rng = np.random.default_rng(params.random_seed + offset + 1)

    def simulate(self) -> Path:
        """Simulate a single path of the process."""
        jump_times, jump_sizes = self._generate_all_jumps(float(self.params.T))
        time_grid = self._build_time_grid(jump_times)
        jump_map = self._map_jumps_to_grid(time_grid, jump_times, jump_sizes)

        values = self._simulate_along_grid(time_grid, jump_map)

        return Path(times=time_grid, values=values)

    def simulate_many(self, n: int) -> list[Path]:
        """Simulate n independent paths."""
        return [self.simulate() for _ in range(n)]

    def _generate_all_jumps(self, t: float) -> tuple[FloatArray, FloatArray]:
        """Generate all large jumps (positive and negative) up to time t."""
        jump_sets = [self._generate_one_sided_jumps(ctx, t) for ctx in self.jump_contexts]
        return self._merge_jumps(jump_sets)

    @staticmethod
    def _generate_one_sided_jumps(ctx: JumpContext, t: float) -> tuple[FloatArray, FloatArray]:
        """Generate jumps of a single sign (positive or negative)."""
        times = _sample_jump_times(ctx, t)

        if times.size == 0:
            return times, np.empty(0, dtype=float)

        sizes = np.array([_sample_jump_size(ctx, tt) for tt in times], dtype=float)

        return times, sizes

    @staticmethod
    def _merge_jumps(jump_sets: list[tuple[FloatArray, FloatArray]]) -> tuple[FloatArray, FloatArray]:
        """Merge jump sequences from multiple signs without sorting."""
        if not jump_sets:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        filtered = [(times, sizes) for times, sizes in jump_sets if times.size > 0]
        if not filtered:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        all_times = np.concatenate([times for times, _ in filtered])
        all_sizes = np.concatenate([sizes for _, sizes in filtered])

        return all_times, all_sizes

    def _build_time_grid(self, jump_times: FloatArray) -> FloatArray:
        """Create time grid combining uniform grid and jump times."""
        n_points = max(int(self.params.N) + 1, 2)
        regular_times = np.linspace(0.0, self.params.T, n_points, dtype=float)
        return np.union1d(regular_times, jump_times)

    @staticmethod
    def _map_jumps_to_grid(time_grid: FloatArray, jump_times: FloatArray, jump_sizes: FloatArray) -> FloatArray:
        """Map jumps to time grid indices."""
        if jump_times.size == 0:
            return np.zeros_like(time_grid, dtype=float)

        jump_map = np.zeros_like(time_grid, dtype=float)
        indices = np.searchsorted(time_grid, jump_times)
        np.add.at(jump_map, indices, jump_sizes)
        return jump_map

    def _simulate_along_grid(self, time_grid: FloatArray, jump_map: FloatArray) -> FloatArray:
        """Construct the process path along the time grid."""
        values = np.empty_like(time_grid, dtype=float)
        values[0] = current_value = float(self.params.x0)

        for idx in range(1, len(time_grid)):
            current_value = self._advance_state(
                t_prev=float(time_grid[idx - 1]),
                t_curr=float(time_grid[idx]),
                x_prev=current_value,
                jump_size=float(jump_map[idx]),
            )
            values[idx] = current_value

        return values

    def _advance_state(self, t_prev: float, t_curr: float, x_prev: float, jump_size: float) -> float:
        """Compute the process value at the next time step."""
        x_next = x_prev

        x_next += self._apply_drift(x_prev, t_prev, t_curr)
        x_next += self._apply_diffusion(x_prev, t_prev, t_curr)

        if jump_size != 0.0:
            x_next += self._apply_large_jump(t_curr, x_prev, jump_size)

        if self.approximate_small_jumps:
            x_next += self._apply_small_jumps(x_prev, t_prev, t_curr)

        if self.compensate_large_jumps:
            x_next -= self._apply_compensator(x_prev, t_prev, t_curr)

        return x_next

    def _apply_drift(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Apply drift increment."""
        drift = self.sde.drift_coefficient
        return drift(t_prev, x_prev) * (t_curr - t_prev)

    def _apply_diffusion(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Apply diffusion increment."""
        rand_norm = self._diffusion_rng.standard_normal()
        diffusion = self.sde.diffusion_coefficient
        return diffusion(t_prev, x_prev) * np.sqrt(t_curr - t_prev) * rand_norm

    def _apply_large_jump(self, t: float, x: float, jump_size: float) -> float:
        """Apply large jump increment."""
        return self.sde.jump_coefficient(t, x, jump_size)

    def _apply_small_jumps(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Apply small-jump Gaussian approximation."""
        variance = self._total_small_jump_variance(x_prev, t_prev, t_curr)
        if variance <= 0.0:
            return 0.0
        return np.sqrt(variance) * self._small_jump_rng.standard_normal()

    def _apply_compensator(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Evaluate the compensator contribution over the step."""
        total = 0.0
        for ctx in self.jump_contexts:
            one_side_comp = ctx.law.compensator(x_prev, t_prev, t_curr, sigma=self.sde.sigma)
            total += float(ctx.sign) * one_side_comp
        return total

    def _total_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Aggregate Gaussian variances contributed by each jump law."""
        total = 0.0
        for ctx in self.jump_contexts:
            total += ctx.law.small_jump_variance(x_prev, t_prev, t_curr, sigma=self.sde.sigma)
        return total


def _sample_jump_times(ctx: JumpContext, t: float) -> FloatArray:
    times = ctx.law.sample_jump_times(ctx.rng, t)
    return np.asarray(times, dtype=float)


def _sample_jump_size(ctx: JumpContext, t: float) -> float:
    return ctx.law.sample_jump_size(ctx.rng, t, ctx.sign)
