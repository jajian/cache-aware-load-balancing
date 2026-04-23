"""Core simulation logic."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from edge_sim.models import Server, SimulationResult, Task
from edge_sim.routing import RoutingStrategy


def simulate(
    tasks: list[Task],
    server_count: int,
    cache_hit_service_time: float,
    strategy: RoutingStrategy,
    workload_name: str,
    arrival_mode: str,
    repetition_label: str,
    repetition_alpha: float,
    trial_seed: int,
    hybrid_probability: float,
    rng: np.random.Generator,
) -> SimulationResult:
    """Simulate a single run with a fixed task list and routing strategy."""
    servers = [Server(server_id=index) for index in range(server_count)]
    completion_times: list[float] = []
    waiting_times: list[float] = []
    cache_hit_count = 0

    for task in tasks:
        server = strategy.select_server(task, servers, rng)
        cache_ready_time = server.completed_cache_ready_time.get(task.task_type)
        is_cache_hit = cache_ready_time is not None and cache_ready_time <= task.arrival_time

        service_time = cache_hit_service_time if is_cache_hit else task.base_service_time
        start_time = max(task.arrival_time, server.next_available_time)
        finish_time = start_time + service_time

        server.next_available_time = finish_time
        server.task_count += 1
        if is_cache_hit:
            server.cache_hits += 1
            cache_hit_count += 1
        elif task.task_type not in server.completed_cache_ready_time:
            # Cache becomes warm only after the first completion of this task type.
            server.completed_cache_ready_time[task.task_type] = finish_time

        completion_times.append(finish_time - task.arrival_time)
        waiting_times.append(start_time - task.arrival_time)

    task_counts = [server.task_count for server in servers]
    makespan = max(server.next_available_time for server in servers) if servers else 0.0
    mean_task_count = float(np.mean(task_counts)) if task_counts else 0.0
    std_task_count = float(np.std(task_counts)) if task_counts else 0.0
    max_task_count = max(task_counts) if task_counts else 0
    min_task_count = min(task_counts) if task_counts else 0
    imbalance_ratio = (max_task_count / mean_task_count) if mean_task_count else 0.0

    distinct_task_types = len({task.task_type for task in tasks})

    return SimulationResult(
        strategy_name=strategy.name,
        workload_name=workload_name,
        arrival_mode=arrival_mode,
        server_count=server_count,
        task_count=len(tasks),
        distinct_task_types=distinct_task_types,
        repetition_label=repetition_label,
        repetition_alpha=repetition_alpha,
        trial_seed=trial_seed,
        hybrid_probability=hybrid_probability,
        makespan=makespan,
        average_completion_time=float(np.mean(completion_times)),
        average_waiting_time=float(np.mean(waiting_times)),
        cache_hit_rate=cache_hit_count / len(tasks) if tasks else 0.0,
        max_server_task_count=max_task_count,
        min_server_task_count=min_task_count,
        mean_server_task_count=mean_task_count,
        std_server_task_count=std_task_count,
        imbalance_ratio=imbalance_ratio,
        max_server_finish_time=makespan,
        per_server_task_counts=task_counts,
        extra={},
    )


def result_to_flat_dict(result: SimulationResult) -> dict[str, object]:
    """Convert a result dataclass to a flat dictionary for CSV export."""
    flat = asdict(result)
    flat["per_server_task_counts"] = ",".join(str(value) for value in result.per_server_task_counts)
    return flat

