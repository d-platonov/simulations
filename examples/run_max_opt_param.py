from pathlib import Path

import numpy as np
from scipy.special import gamma

from levy_type import AdditiveSDE, AlphaStableAR, JumpSign, ProcessSimulator, SimulationParams
from levy_type.numerics import M_VALUES
from examples.plot_utils.moment_accuracy import plot_moment_accuracy


def true_alpha_stable_moment(p: float, alpha: float, t: float, sigma: float = 0.0) -> float:
    """Calculate true p-th moment of a symmetric alpha-stable process (sigma = 0)."""
    power = 1 - sigma * alpha
    time_scale = (t**power / power) ** (p / alpha)
    return time_scale * (2**p * gamma((1 + p) / 2) * gamma(1 - p / alpha)) / (gamma(1 - p / 2) * gamma(1 / 2))


def expected_number_of_jumps(T: float, alpha: float, delta: float) -> float:
    return T * (2 * M_VALUES[alpha]) / (delta**alpha)


def main():
    # Simulation / visualization settings
    T = 0.1
    sigma = 0.0
    alpha = 1.5
    p = 0.3
    delta_list = [0.05, 0.04, 0.03, 0.02, 0.01]
    n_paths = 10_000
    threshold = 0.02
    plot_dir = str(Path(__file__).resolve().parent / "plots")
    random_seed = 2025

    sde = AdditiveSDE(
        drift_coefficient=lambda t, x: 0.0,
        diffusion_coefficient=lambda t, x: 0.0,
        sigma=sigma,
    )
    params = SimulationParams(T=T, N=0, x0=0.0, random_seed=random_seed)

    true_moment = true_alpha_stable_moment(p=p, alpha=alpha, t=T, sigma=sigma)

    results = []
    for delta in delta_list:
        simulator_ar_alpha_stable = ProcessSimulator(
            sde=sde,
            jump_law={
                JumpSign.POSITIVE: AlphaStableAR(alpha=alpha, delta=delta),
                JumpSign.NEGATIVE: AlphaStableAR(alpha=alpha, delta=delta),
            },
            params=params,
            approximate_small_jumps=False,
            compensate_large_jumps=False,
        )

        paths_ar_alpha_stable = simulator_ar_alpha_stable.simulate_many(n=n_paths)
        ar_samples = np.array([path.values[-1] for path in paths_ar_alpha_stable])
        estimated_moment = np.mean(np.abs(ar_samples) ** p)
        abs_diff = abs(true_moment - estimated_moment)
        expected_jumps = expected_number_of_jumps(T, alpha, delta)
        results.append(
            {
                "delta": delta,
                "estimated_moment": estimated_moment,
                "abs_diff": abs_diff,
                "expected_jumps": expected_jumps,
            }
        )

        print(
            f"delta={delta:.5f} | E[|X_T|^{p}] true={true_moment:.6f} est={estimated_moment:.6f} "
            f"|diff|={abs_diff:.6e} expected_jumps={expected_jumps:.1f}"
        )

    fig = plot_moment_accuracy(
        results=results,
        threshold=threshold,
        delta_list=delta_list,
        alpha=alpha,
        p=p,
        T=T,
        n_paths=n_paths,
        plot_dir=plot_dir,
    )

    print(f"Saved plot to {fig}")


if __name__ == "__main__":
    main()
