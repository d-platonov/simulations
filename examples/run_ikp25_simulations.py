from __future__ import annotations

import numpy as np

from levy_type import (
    AdditiveSDE,
    AlphaStableAR,
    AlphaStableDC,
    GammaAR,
    GammaDC,
    ProcessSimulator,
    SimulationParams,
    JumpSign,
)


def main() -> None:
    additive_sde = AdditiveSDE(drift_coefficient=lambda t, x: 0.0, diffusion_coefficient=lambda t, x: 0.0, sigma=0.0)
    simulation_params = SimulationParams(T=1.0, N=0, x0=0.0, random_seed=2025)

    simulator_ar_alpha_stable = ProcessSimulator(
        sde=additive_sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableAR(alpha=1.0, delta=1e-2),
            JumpSign.NEGATIVE: AlphaStableAR(alpha=1.0, delta=1e-2),
        },
        params=simulation_params,
        approximate_small_jumps=True,
        compensate_large_jumps=True,
    )
    paths_ar_alpha_stable = simulator_ar_alpha_stable.simulate_many(n=1000)
    est = np.mean([abs(path.values[-1]) ** 0.3 for path in paths_ar_alpha_stable])
    print(f"True E[|X_T|^0.3]: {1.122326}")
    print(f"Estimated E[|X_T|^0.3] (AR): {est}")

    simulator_dc_alpha_stable = ProcessSimulator(
        sde=additive_sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableDC(alpha=1.0, h=5e-15, eps=0.125),
            JumpSign.NEGATIVE: AlphaStableDC(alpha=1.0, h=5e-15, eps=0.125),
        },
        params=simulation_params,
        approximate_small_jumps=True,
        compensate_large_jumps=False,
    )
    paths_dc_alpha_stable = simulator_dc_alpha_stable.simulate_many(n=1000)
    est = np.mean([abs(path.values[-1]) ** 0.3 for path in paths_dc_alpha_stable])
    print(f"Estimated E[|X_T|^0.3] (DC): {est}")

    simulator_ar = ProcessSimulator(
        sde=additive_sde,
        jump_law={JumpSign.POSITIVE: GammaAR(gamma=5, lam=2, delta=0.001)},
        params=simulation_params,
        approximate_small_jumps=False,
        compensate_large_jumps=False,
    )
    paths_ar = simulator_ar.simulate_many(n=1000)
    # plot_paths(paths_ar, title="Gamma AR Process Paths")
    true_mean = 5 / 2
    print(f"True Gamma Process Mean: {true_mean}")
    print(f"Estimated Gamma Process Mean (AR): {np.mean([path.values[-1] for path in paths_ar])}")

    simulator_dc = ProcessSimulator(
        sde=additive_sde,
        jump_law={JumpSign.POSITIVE: GammaDC(gamma=5, lam=2, h=1e-14, eps=0.1)},
        params=simulation_params,
        approximate_small_jumps=False,
        compensate_large_jumps=False,
    )
    paths_dc = simulator_dc.simulate_many(n=1000)
    # plot_paths(paths_dc, title="Gamma DC Process Paths")
    print(f"Estimated Gamma Process Mean (DC): {np.mean([path.values[-1] for path in paths_dc])}")


if __name__ == "__main__":
    main()
