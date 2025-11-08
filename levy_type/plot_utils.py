from matplotlib import pyplot as plt

from levy_type.process_simulator import Path


def plot_paths(
    paths: Path | list[Path],
    title: str = "Simulated Process Paths",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> None:
    if not isinstance(paths, list):
        paths = [paths]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, path in enumerate(paths):
        ax.plot(path.times, path.values, drawstyle='steps-post', label=f'Path {i + 1}', alpha=0.8, linewidth=1.5)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
