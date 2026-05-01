"""Routing strategies for choosing candidate servers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from edge_sim.models import Server, Task
from edge_sim.utils import server_score


def _choose_lower_load(
    server_a: Server, server_b: Server, arrival_time: float, rng: np.random.Generator
) -> Server:
    """Pick the less loaded server, using random tie-breaking when loads match."""
    load_a = server_a.queued_work(arrival_time)
    load_b = server_b.queued_work(arrival_time)
    if load_a < load_b:
        return server_a
    if load_b < load_a:
        return server_b
    return server_a if rng.random() < 0.5 else server_b


@dataclass
class RoutingStrategy:
    """Base interface for routing strategies."""

    name: str

    def select_server(
        self, task: Task, servers: Sequence[Server], rng: np.random.Generator
    ) -> Server:
        raise NotImplementedError


@dataclass
class PowerOfTwoChoicesStrategy(RoutingStrategy):
    """Sample two servers uniformly at random and choose the lower-load one."""

    def __init__(self) -> None:
        super().__init__(name="power_of_two")

    def select_server(
        self, task: Task, servers: Sequence[Server], rng: np.random.Generator
    ) -> Server:
        server_count = len(servers)
        sample_size = 2 if server_count >= 2 else 1
        indices = rng.choice(server_count, size=sample_size, replace=False)
        if sample_size == 1:
            return servers[int(indices[0])]
        first = servers[int(indices[0])]
        second = servers[int(indices[1])]
        return _choose_lower_load(first, second, task.arrival_time, rng)


@dataclass
class HashBasedLoadAwareStrategy(RoutingStrategy):
    """Use deterministic hashing to pick two candidate servers, then load-balance."""

    def __init__(self) -> None:
        super().__init__(name="hash_load_aware")

    def select_server(
        self, task: Task, servers: Sequence[Server], rng: np.random.Generator
    ) -> Server:
        scored_servers = sorted(
            servers,
            key=lambda server: server_score(task.task_type, server.server_id),
            reverse=True,
        )
        hypothetical_loads = [scored_servers[i].queued_work(task.arrival_time) for i in range(len(scored_servers))]
        total_load = sum(hypothetical_loads)
        start = 0
        while start < len(hypothetical_loads) - 1 and hypothetical_loads[start] > 0.3 * total_load:
            start += 1
        
        first = scored_servers[start]
        second = scored_servers[start + 1] if start + 1 < len(scored_servers) else scored_servers[start]
        return _choose_lower_load(first, second, task.arrival_time, rng)


@dataclass
class HybridEpsilonStrategy(RoutingStrategy):
    """Mix hash-based routing and random power-of-two routing."""

    epsilon_hash_probability: float
    power_of_two: PowerOfTwoChoicesStrategy = field(init=False)
    hash_based: HashBasedLoadAwareStrategy = field(init=False)

    def __init__(self, epsilon_hash_probability: float) -> None:
        self.epsilon_hash_probability = epsilon_hash_probability
        self.power_of_two = PowerOfTwoChoicesStrategy()
        self.hash_based = HashBasedLoadAwareStrategy()
        super().__init__(name=f"hybrid_p_{epsilon_hash_probability:.2f}")

    def select_server(
        self, task: Task, servers: Sequence[Server], rng: np.random.Generator
    ) -> Server:
        if rng.random() < self.epsilon_hash_probability:
            return self.hash_based.select_server(task, servers, rng)
        return self.power_of_two.select_server(task, servers, rng)
