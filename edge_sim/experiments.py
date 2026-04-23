"""Experiment orchestration, CSV export, and plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from edge_sim.plotting import plot_hybrid_sweeps, plot_repetition_sweeps
from edge_sim.routing import (
    HashBasedLoadAwareStrategy,
    HybridEpsilonStrategy,
    PowerOfTwoChoicesStrategy,
)
from edge_sim.simulator import result_to_flat_dict, simulate
from edge_sim.workload import WorkloadProfile, generate_workload, get_profiles


def load_config(path: Path) -> dict:
    """Load a JSON configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_output_dirs(base_output_dir: Path) -> tuple[Path, Path]:
    """Create output directories for tables and plots."""
    csv_dir = base_output_dir / "csv"
    plots_dir = base_output_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, plots_dir


def build_trial_seeds(base_seed: int, num_trials: int) -> list[int]:
    """Derive deterministic trial seeds from one base seed."""
    rng = np.random.default_rng(base_seed)
    return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=num_trials)]


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over trials for each experiment setting."""
    metric_columns = [
        "makespan",
        "average_completion_time",
        "average_waiting_time",
        "cache_hit_rate",
        "max_server_task_count",
        "min_server_task_count",
        "mean_server_task_count",
        "std_server_task_count",
        "imbalance_ratio",
        "max_server_finish_time",
    ]
    group_columns = [
        "strategy_name",
        "strategy_label",
        "workload_name",
        "arrival_mode",
        "server_count",
        "task_count",
        "distinct_task_types",
        "repetition_label",
        "repetition_alpha",
        "hybrid_probability",
    ]
    return raw_results.groupby(group_columns, as_index=False)[metric_columns].mean()


def run_experiments(config: dict) -> None:
    """Run all configured simulations and write CSV + plot outputs."""
    output_dir = Path(config["output_dir"])
    csv_dir, plots_dir = ensure_output_dirs(output_dir)

    profiles = get_profiles(config["workload_profiles"])
    trial_seeds = build_trial_seeds(config["random_seed"], config["num_trials"])
    default_hybrid_probability = float(config["default_hybrid_probability"])
    hybrid_sweep_probabilities = [float(value) for value in config["hybrid_probabilities"]]

    raw_rows: list[dict[str, object]] = []

    for arrival_mode in config["arrival_modes"]:
        for server_count in config["server_counts"]:
            for profile in profiles:
                for trial_seed in trial_seeds:
                    workload_rng = np.random.default_rng(trial_seed)
                    tasks = generate_workload(
                        num_tasks=config["num_tasks"],
                        num_task_types=config["num_task_types"],
                        arrival_mode=arrival_mode,
                        cache_miss_service_time=config["cache_miss_service_time"],
                        profile=profile,
                        rng=workload_rng,
                        arrival_rate=config.get("arrival_rate"),
                    )

                    base_strategies = [
                        (PowerOfTwoChoicesStrategy(), 0.0, "power_of_two"),
                        (HashBasedLoadAwareStrategy(), 1.0, "hash_load_aware"),
                        (
                            HybridEpsilonStrategy(default_hybrid_probability),
                            default_hybrid_probability,
                            f"hybrid(p={default_hybrid_probability:.2f})",
                        ),
                    ]

                    for strategy, hybrid_probability, strategy_label in base_strategies:
                        strategy_rng = np.random.default_rng(trial_seed + int(hybrid_probability * 1000) + 17)
                        result = simulate(
                            tasks=tasks,
                            server_count=server_count,
                            cache_hit_service_time=config["cache_hit_service_time"],
                            strategy=strategy,
                            workload_name=profile.name,
                            arrival_mode=arrival_mode,
                            repetition_label=profile.repetition_label,
                            repetition_alpha=profile.alpha,
                            trial_seed=trial_seed,
                            hybrid_probability=hybrid_probability,
                            rng=strategy_rng,
                        )
                        row = result_to_flat_dict(result)
                        row["strategy_label"] = strategy_label
                        row["experiment_group"] = "baseline"
                        raw_rows.append(row)

                    for hybrid_probability in hybrid_sweep_probabilities:
                        strategy_rng = np.random.default_rng(trial_seed + int(hybrid_probability * 1000) + 103)
                        hybrid_result = simulate(
                            tasks=tasks,
                            server_count=server_count,
                            cache_hit_service_time=config["cache_hit_service_time"],
                            strategy=HybridEpsilonStrategy(hybrid_probability),
                            workload_name=profile.name,
                            arrival_mode=arrival_mode,
                            repetition_label=profile.repetition_label,
                            repetition_alpha=profile.alpha,
                            trial_seed=trial_seed,
                            hybrid_probability=hybrid_probability,
                            rng=strategy_rng,
                        )
                        hybrid_row = result_to_flat_dict(hybrid_result)
                        hybrid_row["strategy_label"] = f"hybrid(p={hybrid_probability:.2f})"
                        hybrid_row["experiment_group"] = "hybrid_sweep"
                        raw_rows.append(hybrid_row)

    raw_results = pd.DataFrame(raw_rows)
    aggregated_results = summarize_results(raw_results)
    baseline_aggregated = aggregated_results[
        aggregated_results["strategy_label"].isin(
            [
                "power_of_two",
                "hash_load_aware",
                f"hybrid(p={default_hybrid_probability:.2f})",
            ]
        )
    ].copy()
    hybrid_aggregated = aggregated_results[
        aggregated_results["strategy_name"].str.startswith("hybrid_p_")
    ].copy()

    raw_results.to_csv(csv_dir / "raw_results.csv", index=False)
    aggregated_results.to_csv(csv_dir / "aggregated_results.csv", index=False)
    baseline_aggregated.to_csv(csv_dir / "baseline_summary.csv", index=False)
    hybrid_aggregated.to_csv(csv_dir / "hybrid_summary.csv", index=False)

    plot_repetition_sweeps(baseline_aggregated, plots_dir)
    plot_hybrid_sweeps(hybrid_aggregated, plots_dir)

    print(f"Wrote CSV files to: {csv_dir}")
    print(f"Wrote plot files to: {plots_dir}")

