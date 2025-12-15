import numpy as np
from matplotlib import pyplot as plt

from examples.simulation_utils.alpha_stable_utils import get_delta_ar, get_eps_dc, get_h_dc
from levy_type import (
    AdditiveSDE,
    AlphaStableAR,
    AlphaStableDC,
    JumpSign,
    ProcessSimulator,
    SimulationParams,
)


def extract_jump_times(paths: list, t_final: float) -> np.ndarray:
    """Return concatenated jump times, excluding the initial time and horizon."""
    jump_times = [p.times[(p.times > 0.0) & (p.times < t_final)] for p in paths]
    return np.concatenate(jump_times) if jump_times else np.array([], dtype=float)


def plot_jump_time_histograms(times_ar: np.ndarray, times_dc: np.ndarray, t_final: float) -> None:
    """Visualize jump time distributions for AR and DC side by side."""
    plt.style.use("seaborn-v0_8-whitegrid")
    bins = np.linspace(0.0, t_final, 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].hist(times_ar, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.85)
    axes[0].set_title("AR jump times", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Jump count")

    axes[1].hist(times_dc, bins=bins, color="#d62728", edgecolor="white", alpha=0.85)
    axes[1].set_title("DC jump times", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Time")

    fig.suptitle("Histogram of jump times", fontsize=16, fontweight="bold")
    fig.tight_layout()
    plt.show()


def main():
    # Simulation settings
    T = 1.0
    sigma = 0.25
    alpha = 1.5
    n_paths = 1_000
    random_seed = 2025

    # AR / DC settings
    NJ = 500  # Expected number of jumps per path
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
        approximate_small_jumps=True,
        compensate_large_jumps=False,
    )

    paths_ar = simulator_ar.simulate_many(n=n_paths)
    jump_times_ar = extract_jump_times(paths_ar, T)

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

    paths_dc = simulator_dc.simulate_many(n=n_paths)
    jump_times_dc = extract_jump_times(paths_dc, T)

    plot_jump_time_histograms(jump_times_ar, jump_times_dc, T)


if __name__ == "__main__":
    main()
