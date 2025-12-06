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


def combine_paths(paths: list[Path]) -> Path:
    """
    Combines a list of Path objects into a single Path by taking the
    union of all times and the average of values at each time point.
    """
    all_times = np.concatenate([p.times for p in paths])
    unified_times = np.unique(all_times)

    total_values = np.zeros_like(unified_times, dtype=np.float64)

    for p in paths:
        indices = np.searchsorted(p.times, unified_times, side='right') - 1
        indices = np.maximum(indices, 0)
        total_values += p.values[indices]

    mean_values = total_values / len(paths)

    return Path(times=unified_times, values=mean_values)


def main():
    # Simulation settings
    T = 1.0
    sigma = 0.25
    alpha = 1.5
    n_paths = 1_000
    random_seed = 2025

    # Weak error settings
    p = 0.3
    func = lambda x: np.abs(x) ** p

    # AR / DC settings
    NJ = 100
    delta = get_delta_ar(n_jumps=NJ, alpha=alpha, t=T)
    eps = get_eps_dc(alpha=alpha, sigma=sigma)
    h = get_h_dc(n_jumps=NJ, eps_dc=eps, t=T)

    sde = AdditiveSDE(
        drift_coefficient=lambda t, x: 0.0,
        diffusion_coefficient=lambda t, x: 0.0,
        sigma=sigma,
    )
    params = SimulationParams(T=T, N=0, x0=0.0, random_seed=random_seed)

    simulator_ar = ProcessSimulator(
        sde=sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableAR(alpha=alpha, delta=delta),
            JumpSign.NEGATIVE: AlphaStableAR(alpha=alpha, delta=delta),
        },
        params=params,
        approximate_small_jumps=False,
        compensate_large_jumps=False,
    )

    paths_ar = simulator_ar.simulate_many(n=n_paths)

    # Apply Weak Error Function
    for path in paths_ar:
        path.values = func(path.values)

    combined_path_ar = combine_paths(paths_ar)
    true_path = Path(
        times=combined_path_ar.times,
        values=true_alpha_stable_moment(p=p, alpha=alpha, t=combined_path_ar.times, sigma=sigma),
    )
    plot_paths(
        [combined_path_ar, true_path],
        title="Cumulative Weak Error for Alpha-Stable Process (AR)",
        labels=["Estimated", "True"],
    )

    simulator_dc = ProcessSimulator(
        sde=sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableDC(alpha=alpha, h=h, eps=eps),
            JumpSign.NEGATIVE: AlphaStableDC(alpha=alpha, h=h, eps=eps),
        },
        params=params,
        approximate_small_jumps=False,
        compensate_large_jumps=False,
    )

    paths_dc = simulator_dc.simulate_many(n=n_paths)

    # Apply Weak Error Function
    for path in paths_dc:
        path.values = func(path.values)

    combined_path_dc = combine_paths(paths_dc)
    plot_paths(
        [combined_path_dc, true_path],
        title="Cumulative Weak Error for Alpha-Stable Process (DC)",
        labels=["Estimated", "True"],
    )


if __name__ == "__main__":
    main()
