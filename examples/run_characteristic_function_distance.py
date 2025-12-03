import numpy as np
from matplotlib import pyplot as plt

from levy_type import AdditiveSDE, AlphaStableAR, AlphaStableDC, JumpSign, ProcessSimulator, SimulationParams
from levy_type.numerics import M_VALUES


def theoretical_cf(xi: np.ndarray, alpha: float, sigma: float, t: float = 1.0) -> np.ndarray:
    """Theoretical characteristic function (CF) for symmetric alpha-stable process."""
    scale = (t ** (1 - sigma * alpha)) / (1 - sigma * alpha)
    return np.exp(-scale * np.abs(xi) ** alpha)


def empirical_cf(xi: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Compute empirical CF for a vector of samples: phi_hat(xi) = (1/N) * sum(exp(i * xi * samples))."""
    exponents = np.exp(1j * samples[:, np.newaxis] * xi[np.newaxis, :])
    return np.mean(exponents, axis=0)


def compute_l2_distance(samples: np.ndarray, alpha: float, sigma: float, t: float = 1.0):
    """
    Compute the weighted L2 distance defined by |phi_hat - phi_true|^2 * Gaussian_PDF dxi
    We approximate the integral using a discrete grid of xi within the range [-4, 4] (covers 99.9%).
    """
    xi_min, xi_max = -5.0, 5.0
    n_points = 500
    xi_grid = np.linspace(xi_min, xi_max, n_points)
    d_xi = xi_grid[1] - xi_grid[0]

    phi_hat = empirical_cf(xi=xi_grid, samples=samples)
    phi_true = theoretical_cf(xi=xi_grid, alpha=alpha, sigma=sigma, t=t)

    diff_sq = np.abs(phi_hat - phi_true) ** 2
    pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xi_grid**2)
    dist = np.sum(diff_sq * pdf) * d_xi

    return dist, xi_grid, phi_hat, phi_true


def get_delta_ar(n_jumps: float, alpha: float, t: float = 1.0):
    m = M_VALUES[alpha]
    return (2 * m * t / n_jumps) ** (1 / alpha)


def get_eps_dc(alpha: float, sigma: float):
    return alpha * sigma


def get_h_dc(n_jumps: float, eps_dc: float, t: float = 1.0):
    return ((2 * t ** (1 - eps_dc)) / (n_jumps * (1 - eps_dc))) ** (1 / eps_dc)


def main():
    sigma = 0
    seed = 42
    T = 1.0
    sde = AdditiveSDE(drift_coefficient=lambda t, x: 0.0, diffusion_coefficient=lambda t, x: 0.0, sigma=sigma)
    params = SimulationParams(T=T, N=0, x0=0.0, random_seed=seed)

    alpha = 1.5
    NJ = 250
    delta = get_delta_ar(n_jumps=NJ, alpha=alpha, t=params.T)
    eps = get_eps_dc(alpha=alpha, sigma=sigma) if sigma > 0 else 0.1
    h = get_h_dc(n_jumps=NJ, eps_dc=eps, t=params.T)

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

    simulator_dc_alpha_stable = ProcessSimulator(
        sde=sde,
        jump_law={
            JumpSign.POSITIVE: AlphaStableDC(alpha=alpha, h=h, eps=eps),
            JumpSign.NEGATIVE: AlphaStableDC(alpha=alpha, h=h, eps=eps),
        },
        params=params,
        approximate_small_jumps=False,
        compensate_large_jumps=False,
    )

    paths_ar_alpha_stable = simulator_ar_alpha_stable.simulate_many(n=50_000)
    paths_dc_alpha_stable = simulator_dc_alpha_stable.simulate_many(n=50_000)

    ar_samples = np.array([path.values[-1] for path in paths_ar_alpha_stable])
    dc_samples = np.array([path.values[-1] for path in paths_dc_alpha_stable])

    # --- Compute Distances ---
    dist_ar, xi_grid, phi_ar, phi_true_vals = compute_l2_distance(samples=ar_samples, alpha=alpha, sigma=sigma, t=T)
    dist_dc, _, phi_dc, _ = compute_l2_distance(samples=dc_samples, alpha=alpha, sigma=sigma, t=T)

    print(f"Weighted L2 Distance (AR): {dist_ar:.8f}")
    print(f"Weighted L2 Distance (DC): {dist_dc:.8f}")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))

    plt.plot(xi_grid, phi_true_vals.real, 'k-', linewidth=2, label=r'True $\phi^Z(\xi) = e^{-|\xi|^\alpha}$')
    plt.plot(xi_grid, phi_ar.real, 'r--', label=f'AR Empirical $\hat{{\phi}}$ (Dist: {dist_ar:.5f})')
    plt.plot(xi_grid, phi_dc.real, 'b-.', label=f'DC Empirical $\hat{{\phi}}$ (Dist: {dist_dc:.5f})')

    plt.title('Comparison of Characteristic Functions Weighted $L^2$ Distance')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$Re(\phi(\xi))$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)

    plt.savefig(f"./plots/cf_distance_comparison_sigma_{sigma}_T_{T}_seed_{seed}.png")


if __name__ == "__main__":
    main()
