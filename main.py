"""Command-line entry point for the edge-computing task-routing simulator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

from edge_sim.experiments import load_config, run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run randomized task-assignment experiments for an edge-computing system."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/example_config.json"),
        help="Path to the JSON experiment configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_experiments(config)


if __name__ == "__main__":
    main()
