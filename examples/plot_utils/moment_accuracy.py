import os
from typing import Iterable, List

from matplotlib import pyplot as plt


def _format_float_for_filename(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def plot_moment_accuracy(
    *,
    results: List[dict],
    threshold: float,
    delta_list: Iterable[float],
    alpha: float,
    p: float,
    T: float,
    n_paths: int,
    plot_dir: str,
) -> str:
    """Plot accuracy and expected jumps against delta and save the figure."""

    sorted_results = sorted(results, key=lambda res: res["delta"])
    deltas = [res["delta"] for res in sorted_results]
    abs_diffs = [res["abs_diff"] for res in sorted_results]
    expected_jumps = [res["expected_jumps"] for res in sorted_results]

    fig, (ax_err, ax_jumps) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    ax_err.plot(deltas, abs_diffs, marker="o", linewidth=2, color="#1f77b4", label="|true - estimated|")
    ax_err.fill_between(
        deltas,
        abs_diffs,
        threshold,
        where=[diff <= threshold for diff in abs_diffs],
        color="#1f77b4",
        alpha=0.15,
    )
    ax_err.axhline(threshold, color="#d62728", linestyle="--", label=f"threshold={threshold}")
    ax_err.set_ylabel("Absolute error")
    ax_err.set_title("Moment accuracy / effort vs. delta")
    ax_err.grid(True, axis="y", alpha=0.3)

    if not any(diff <= threshold for diff in abs_diffs):
        ax_err.text(
            0.5,
            0.85,
            "No delta meets threshold",
            transform=ax_err.transAxes,
            ha="center",
            color="#d62728",
            fontsize=9,
        )

    ax_jumps.plot(deltas, expected_jumps, marker="s", color="#111111", linewidth=1.8, label="expected jumps")
    ax_jumps.set_ylabel("Expected jumps")
    ax_jumps.set_xlabel("delta")
    ax_jumps.grid(True, axis="y", alpha=0.3)

    ax_jumps.set_xticks(deltas)
    ax_jumps.set_xticklabels([f"{delta:.3f}" for delta in deltas])

    ax_err.legend(loc="upper left")
    ax_jumps.legend(loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(plot_dir, exist_ok=True)
    deltas_str = "-".join(_format_float_for_filename(delta) for delta in delta_list)
    plot_name = (
        f"moment_accuracy_alpha{_format_float_for_filename(alpha)}_p{_format_float_for_filename(p)}"
        f"_T{_format_float_for_filename(T)}_paths{n_paths}_thr{_format_float_for_filename(threshold)}"
        f"_deltas{deltas_str}.png"
    )
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    return plot_path
