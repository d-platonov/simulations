from levy_type.base_process import DEFAULT_SIGMA
from levy_type.constants import M_VALUES
from levy_type.plot_utils import plot_paths
from levy_type.process_simulator import ProcessSimulator
from levy_type.simulation_config import ARSimulationConfig, DCSimulationConfig


def _get_delta_ar(n_jumps: float, alpha: float, t: float = 1.0):
    m = M_VALUES[alpha]
    return (2 * m * t / n_jumps) ** (1 / alpha)


def _get_eps_dc(alpha: float, sigma: float):
    if sigma == 0.0:
        return 0.1  # Use fixed eps=0.1 as per Section 8.3
    return alpha * sigma


def _get_h_dc(n_jumps: float, eps_dc: float, t: float = 1.0):
    return ((2 * t ** (1 - eps_dc)) / (n_jumps * (1 - eps_dc))) ** (1 / eps_dc)


def main():
    alpha = 0.5

    config_ar = ARSimulationConfig(
        T=1.0,
        N=0,
        delta=_get_delta_ar(n_jumps=1000, alpha=0.5),
        x_0=0.0,
        alpha=alpha,
        random_seed=2025,
    )
    simulator_ar = ProcessSimulator(config=config_ar, approximate_small_jumps=False)
    paths_ar = simulator_ar.simulate_many(n=100)
    plot_paths(paths_ar)

    eps_dc = _get_eps_dc(alpha=alpha, sigma=DEFAULT_SIGMA)
    config_dc = DCSimulationConfig(
        T=1.0,
        N=0,
        h=_get_h_dc(n_jumps=1000, eps_dc=eps_dc),
        eps=eps_dc,
        x_0=0.0,
        alpha=0.5,
        random_seed=2025,
    )
    simulator_dc = ProcessSimulator(config=config_dc, approximate_small_jumps=False)
    paths_dc = simulator_dc.simulate_many(n=100)
    plot_paths(paths_dc)


if __name__ == "__main__":
    main()
