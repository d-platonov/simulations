import numpy as np
from scipy.special import gamma

from levy_type import AdditiveSDE, AlphaStableAR, JumpSign, ProcessSimulator, SimulationParams
from levy_type.numerics import M_VALUES


def true_alpha_stable_moment(p: float, alpha: float, t: float, sigma: float = 0.0) -> float:
    """Calculate true p-th moment of a symmetric alpha-stable process (sigma = 0)."""
    power = 1 - sigma * alpha
    time_scale = (t**power / power) ** (p / alpha)
    return time_scale * (2**p * gamma((1 + p) / 2) * gamma(1 - p / alpha)) / (gamma(1 - p / 2) * gamma(1 / 2))


def main():
    T = 0.1
    sigma = 0
    sde = AdditiveSDE(drift_coefficient=lambda t, x: 0.0, diffusion_coefficient=lambda t, x: 0.0, sigma=sigma)
    params = SimulationParams(T=T, N=0, x0=0.0, random_seed=42)

    alpha = 1.5
    delta = 0.01

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

    paths_ar_alpha_stable = simulator_ar_alpha_stable.simulate_many(n=1_000)

    ar_samples = np.array([path.values[-1] for path in paths_ar_alpha_stable])

    p = 0.3
    true_moment = true_alpha_stable_moment(p=p, alpha=alpha, t=T, sigma=sigma)
    estimated_moment = np.mean(np.abs(ar_samples) ** p)

    print(f"True E[|X_T|^{p}]: {true_moment}")
    print(f"Estimated E[|X_T|^{p}] (AR): {estimated_moment}")
    print(f"Expected number of jumps: {T * (2 * M_VALUES[alpha]) / (delta**alpha)}")


if __name__ == "__main__":
    main()
