"""Plot experiment outputs with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_METRICS = {
    "makespan": "Makespan",
    "average_completion_time": "Average Completion Time",
    "cache_hit_rate": "Cache Hit Rate",
    "max_server_task_count": "Max Server Load (Task Count)",
}


def _sanitize_filename(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def plot_repetition_sweeps(aggregated: pd.DataFrame, output_dir: Path) -> None:
    """Create metric-vs-repetition plots for each arrival mode and server count."""
    for (arrival_mode, server_count), subset in aggregated.groupby(["arrival_mode", "server_count"]):
        strategies = subset["strategy_label"].unique()
        x_labels = list(dict.fromkeys(subset["repetition_label"].tolist()))
        for metric, y_label in PLOT_METRICS.items():
            plt.figure(figsize=(8, 5))
            for strategy in strategies:
                strategy_subset = subset[subset["strategy_label"] == strategy].copy()
                strategy_subset["repetition_label"] = pd.Categorical(
                    strategy_subset["repetition_label"], categories=x_labels, ordered=True
                )
                strategy_subset = strategy_subset.sort_values("repetition_label")
                plt.plot(
                    strategy_subset["repetition_label"].astype(str),
                    strategy_subset[metric],
                    marker="o",
                    label=strategy,
                )
            plt.xlabel("Repetition Level")
            plt.ylabel(y_label)
            plt.title(f"{y_label} vs Repetition Level ({arrival_mode}, m={server_count})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            filename = (
                f"{metric}_{_sanitize_filename(arrival_mode)}_m{server_count}.png"
            )
            plt.savefig(output_dir / filename, dpi=160)
            plt.close()


def plot_hybrid_sweeps(raw_hybrid: pd.DataFrame, output_dir: Path) -> None:
    """Create plots showing hybrid performance as p changes. Completion time uses boxplots."""
    if raw_hybrid.empty:
        return

    hybrid_metrics = {
        "makespan": "Makespan",
        "average_completion_time": "Average Completion Time",
        "cache_hit_rate": "Cache Hit Rate",
    }

    for (arrival_mode, server_count, repetition_label), subset in raw_hybrid.groupby(
        ["arrival_mode", "server_count", "repetition_label"]
    ):
        for metric, y_label in hybrid_metrics.items():
            plt.figure(figsize=(8, 5))
            
            if metric == "average_completion_time":
                # Create boxplot for each unique p
                probs = sorted(subset["hybrid_probability"].unique())
                data_to_plot = [subset[subset["hybrid_probability"] == p][metric].values for p in probs]
                plt.boxplot(data_to_plot, labels=[f"{p:.2f}" for p in probs], showmeans=False)
                plt.xlabel("Hybrid Probability p")
            else:
                # Use line plot of means for other metrics
                means = subset.groupby("hybrid_probability")[metric].mean().reset_index().sort_values("hybrid_probability")
                plt.plot(
                    means["hybrid_probability"],
                    means[metric],
                    marker="o",
                    label=f"{repetition_label}",
                )
                plt.xlabel("Hybrid Probability p")

            plt.ylabel(y_label)
            plt.title(
                f"{y_label} vs p ({arrival_mode}, m={server_count}, {repetition_label})"
            )
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            filename = (
                f"hybrid_{metric}_{_sanitize_filename(arrival_mode)}_m{server_count}_"
                f"{_sanitize_filename(repetition_label)}.png"
            )
            plt.savefig(output_dir / filename, dpi=160)
            plt.close()

