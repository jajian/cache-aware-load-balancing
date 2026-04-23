"""Data models used by the simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """A single incoming task."""

    task_id: int
    arrival_time: float
    task_type: str
    base_service_time: float


@dataclass
class Server:
    """A single FIFO server tracked by its next available time."""

    server_id: int
    next_available_time: float = 0.0
    completed_cache_ready_time: dict[str, float] = field(default_factory=dict)
    task_count: int = 0
    cache_hits: int = 0

    def queued_work(self, arrival_time: float) -> float:
        """Return how much work is currently queued when a task arrives."""
        return max(0.0, self.next_available_time - arrival_time)


@dataclass
class SimulationResult:
    """Aggregated metrics for a single simulation run."""

    strategy_name: str
    workload_name: str
    arrival_mode: str
    server_count: int
    task_count: int
    distinct_task_types: int
    repetition_label: str
    repetition_alpha: float
    trial_seed: int
    hybrid_probability: float
    makespan: float
    average_completion_time: float
    average_waiting_time: float
    cache_hit_rate: float
    max_server_task_count: int
    min_server_task_count: int
    mean_server_task_count: float
    std_server_task_count: float
    imbalance_ratio: float
    max_server_finish_time: float
    per_server_task_counts: list[int]
    extra: dict[str, Any] = field(default_factory=dict)

