from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from matplotlib import pyplot as plt

from levy_type.simulation.path import Path

__all__: Final = ["plot_paths"]
_STYLE: Final = "seaborn-v0_8-whitegrid"


def _normalize_paths(paths: Path | Sequence[Path]) -> list[Path]:
    return list(paths) if isinstance(paths, Sequence) else [paths]


def plot_paths(
    paths: Path | Sequence[Path],
    title: str = "Simulated Process Paths",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> None:
    """Plot one or many simulated paths using a step-style line plot."""
    path_list = _normalize_paths(paths)

    plt.style.use(_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, path in enumerate(path_list, start=1):
        ax.plot(path.times, path.values, drawstyle="steps-post", label=f"Path {idx}", alpha=0.8, linewidth=1.5)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()
