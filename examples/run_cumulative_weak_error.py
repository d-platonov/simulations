import numpy as np

from examples.simulation_utils.alpha_stable_utils import get_delta_ar, get_eps_dc, get_h_dc, true_alpha_stable_moment
from levy_type import (
    AdditiveSDE,
    AlphaStableAR,
    AlphaStableDC,
    JumpSign,
    Path,
    ProcessSimulator,
    SimulationParams,
    plot_paths,
)


def _values_at_times(path: Path, target_times: np.ndarray) -> np.ndarray:
    """Evaluate a stepwise-constant path at the requested times."""
    indices = np.searchsorted(path.times, target_times, side="right") - 1
    indices = np.clip(indices, 0, len(path.values) - 1)
    return path.values[indices]


def combine_paths(paths: list[Path]) -> Path:
    """
    Combines a list of Path objects into a single Path by taking the
    union of all times and the average of values at each time point.
    """
    all_times = np.concatenate([p.times for p in paths])
    unified_times = np.unique(all_times)

    total_values = np.zeros_like(unified_times, dtype=np.float64)

    for p in paths:
        total_values += _values_at_times(p, unified_times)

    mean_values = total_values / len(paths)

    return Path(times=unified_times, values=mean_values)


def running_sup_abs_error(estimated: Path, truth: Path) -> Path:
    """Running sup_{0<=s<=t} of |estimated - truth| along all jump times."""
    unified_times = np.union1d(estimated.times, truth.times)
    abs_error = np.abs(_values_at_times(estimated, unified_times) - _values_at_times(truth, unified_times))
    return Path(times=unified_times, values=np.maximum.accumulate(abs_error))


def trim_at_plateau(path: Path) -> Path:
    """Keep values up to (and including) the first time the running value hits its final plateau."""
    final_value = path.values[-1]
    hits = np.isclose(path.values, final_value)
    first_hit_idx = int(np.argmax(hits))
    return Path(times=path.times[: first_hit_idx + 1], values=path.values[: first_hit_idx + 1])


def main():
    # Simulation settings
    T = 1.0
    sigma = 0.25
    alpha = 1.5
    n_paths = 1_000
    random_seed = 2025

    # Weak error settings
    p = 0.3
    transform = lambda x: np.abs(x) ** p

    # AR / DC settings
    NJ = 100  # Expected number of jumps per path
    delta = get_delta_ar(n_jumps=NJ, alpha=alpha, t=T)
    eps = get_eps_dc(alpha=alpha, sigma=sigma)
    h = get_h_dc(n_jumps=NJ, eps_dc=eps, t=T)

    sde = AdditiveSDE(
        drift_coefficient=lambda t, x: 0.0,
        diffusion_coefficient=lambda t, x: 0.0,
        sigma=sigma,
    )
    params = SimulationParams(T=T, N=0, x0=0.0, random_seed=random_seed)

    def simulate_mean_path(simulator: ProcessSimulator) -> Path:
        transformed_paths = []
        for path in simulator.simulate_many(n=n_paths):
            transformed_paths.append(Path(times=path.times, values=transform(path.values)))
        return combine_paths(transformed_paths)

    def true_moment_path(times: np.ndarray) -> Path:
        return Path(times=times, values=true_alpha_stable_moment(p=p, alpha=alpha, t=times, sigma=sigma))

    simulator_ar = ProcessSimulator(
        sde=sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableAR(alpha=alpha, delta=delta),
            JumpSign.NEGATIVE: AlphaStableAR(alpha=alpha, delta=delta),
        },
        params=params,
        approximate_small_jumps=True,
        compensate_large_jumps=False,
    )

    mean_path_ar = simulate_mean_path(simulator_ar)
    true_path_ar = true_moment_path(mean_path_ar.times)
    running_error_ar = running_sup_abs_error(mean_path_ar, true_path_ar)

    simulator_dc = ProcessSimulator(
        sde=sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableDC(alpha=alpha, h=h, eps=eps),
            JumpSign.NEGATIVE: AlphaStableDC(alpha=alpha, h=h, eps=eps),
        },
        params=params,
        approximate_small_jumps=True,
        compensate_large_jumps=False,
    )

    mean_path_dc = simulate_mean_path(simulator_dc)
    true_path_dc = true_moment_path(mean_path_dc.times)
    running_error_dc = running_sup_abs_error(mean_path_dc, true_path_dc)

    plot_paths(
        [mean_path_ar, true_path_ar],
        title="Estimated vs. True Moment (AR)",
        labels=["Estimated", "True"],
    )

    plot_paths(
        [mean_path_dc, true_path_dc],
        title="Estimated vs. True Moment (DC)",
        labels=["Estimated", "True"],
    )

    plot_paths(
        [trim_at_plateau(running_error_ar), trim_at_plateau(running_error_dc)],
        title="Running Sup Weak Error",
        labels=["AR", "DC"],
    )


if __name__ == "__main__":
    main()
