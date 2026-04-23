"""Workload generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from edge_sim.models import Task


@dataclass(frozen=True)
class WorkloadProfile:
    """Named workload shape used in experiments."""

    name: str
    repetition_label: str
    alpha: float


WORKLOAD_PROFILES: dict[str, WorkloadProfile] = {
    "mostly_unique": WorkloadProfile(
        name="mostly_unique", repetition_label="mostly_unique", alpha=0.3
    ),
    "moderate_repetition": WorkloadProfile(
        name="moderate_repetition", repetition_label="moderate_repetition", alpha=1.0
    ),
    "highly_repetitive": WorkloadProfile(
        name="highly_repetitive", repetition_label="highly_repetitive", alpha=2.0
    ),
}


def build_task_type_probabilities(num_task_types: int, alpha: float) -> np.ndarray:
    """Build a Zipf-like distribution over task types."""
    ranks = np.arange(1, num_task_types + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


def generate_arrival_times(
    num_tasks: int,
    arrival_mode: str,
    rng: np.random.Generator,
    arrival_rate: float | None = None,
) -> np.ndarray:
    """Generate either simultaneous or stochastic arrival times."""
    if arrival_mode == "all_at_zero":
        return np.zeros(num_tasks, dtype=float)
    if arrival_mode == "poisson":
        if arrival_rate is None or arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive for stochastic arrivals.")
        inter_arrivals = rng.exponential(scale=1.0 / arrival_rate, size=num_tasks)
        return np.cumsum(inter_arrivals)
    raise ValueError(f"Unsupported arrival_mode: {arrival_mode}")


def generate_workload(
    num_tasks: int,
    num_task_types: int,
    arrival_mode: str,
    cache_miss_service_time: float,
    profile: WorkloadProfile,
    rng: np.random.Generator,
    arrival_rate: float | None = None,
) -> list[Task]:
    """Create a list of tasks according to the chosen workload profile."""
    probabilities = build_task_type_probabilities(num_task_types, profile.alpha)
    task_type_indices = rng.choice(num_task_types, size=num_tasks, p=probabilities)
    arrival_times = generate_arrival_times(num_tasks, arrival_mode, rng, arrival_rate)
    tasks = [
        Task(
            task_id=index,
            arrival_time=float(arrival_times[index]),
            task_type=f"type_{int(task_type_indices[index])}",
            base_service_time=cache_miss_service_time,
        )
        for index in range(num_tasks)
    ]
    return sorted(tasks, key=lambda task: (task.arrival_time, task.task_id))


def get_profiles(profile_names: Iterable[str]) -> list[WorkloadProfile]:
    """Resolve named workload profiles from configuration."""
    profiles: list[WorkloadProfile] = []
    for name in profile_names:
        if name not in WORKLOAD_PROFILES:
            raise ValueError(f"Unknown workload profile: {name}")
        profiles.append(WORKLOAD_PROFILES[name])
    return profiles

