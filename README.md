# Edge Task Routing Simulator

This project simulates randomized task assignment in a distributed edge-computing system and compares three routing strategies:

1. `power_of_two`: sample 2 servers uniformly at random and route to the one with less queued work.
2. `hash_load_aware`: deterministically map each `task_type` to 2 candidate servers using a stable hash, then route to the less loaded one.
3. `hybrid`: with probability `p`, use the hash-based rule; with probability `1 - p`, use power-of-two choices.

The simulator is designed to study the tradeoff between load balancing and cache locality.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- main.py
|-- config/
|   `-- example_config.json
|-- edge_sim/
|   |-- __init__.py
|   |-- experiments.py
|   |-- models.py
|   |-- plotting.py
|   |-- routing.py
|   |-- simulator.py
|   |-- utils.py
|   `-- workload.py
`-- example_output/
    |-- csv/
    `-- plots/
```

## System Model

- There are `m` servers, each modeled as a single FIFO queue.
- Each task has `arrival_time`, `task_type`, and `base_service_time`.
- Each server is tracked by `next_available_time`, so the simulator does not need to store full queue contents.
- The load used for routing is:

```text
queued_work = max(0, next_available_time - arrival_time)
```

This means a task observes how much unfinished work is already queued on a server at its arrival time.

## Caching Model

- A server gets a cache benefit for a `task_type` only after it has completed that type at least once.
- If a later task of the same type arrives after that completion time, it is a cache hit and uses `cache_hit_service_time`.
- Otherwise it is a cache miss and uses the full `cache_miss_service_time`.
- The cache is assumed to be persistent for the duration of the experiment and there is no eviction.

This lets the hash-based strategy exploit task-type locality by repeatedly steering identical task types toward the same candidate servers.

## Workloads

The simulator supports two arrival models:

- `all_at_zero`: every task arrives at time 0.
- `poisson`: task arrivals follow exponential inter-arrival times with configurable rate.

It also includes three workload profiles:

- `mostly_unique`: weak skew, so repeated task types are uncommon.
- `moderate_repetition`: moderate Zipf-like skew.
- `highly_repetitive`: strong skew, so a small number of task types dominate.

Internally, repetition is controlled by a Zipf-style parameter `alpha`, and task types are sampled from that distribution.

## Metrics

Each simulation run computes:

- makespan
- average completion time
- average waiting time
- cache hit rate
- per-server task counts
- max server load by task count
- imbalance statistics such as mean, standard deviation, and max/mean ratio

## Experiments

The main experiment runner:

- sweeps over server counts
- sweeps over repetition levels
- evaluates the three primary strategies
- sweeps hybrid probabilities `p` in `{0, 0.25, 0.5, 0.75, 1.0}`
- averages results over multiple random seeds
- writes CSV summaries
- generates matplotlib plots

Generated plots include:

- makespan vs repetition level
- average completion time vs repetition level
- cache hit rate vs repetition level
- max server load vs repetition level
- hybrid performance vs `p`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Run the example experiment:

```bash
python main.py --config config/example_config.json
```

This writes outputs under `example_output/`:

- `example_output/csv/raw_results.csv`: one row per trial and strategy.
- `example_output/csv/aggregated_results.csv`: averaged metrics across trials.
- `example_output/csv/baseline_summary.csv`: summary for the 3 main strategies.
- `example_output/csv/hybrid_summary.csv`: summary for the `p` sweep.
- `example_output/plots/*.png`: generated figures.

## Configuration

The example configuration file shows all major knobs:

- `num_tasks`
- `num_task_types`
- `server_counts`
- `arrival_modes`
- `arrival_rate`
- `workload_profiles`
- `cache_hit_service_time`
- `cache_miss_service_time`
- `default_hybrid_probability`
- `hybrid_probabilities`
- `num_trials`
- `random_seed`
- `output_dir`

You can copy `config/example_config.json`, modify values, and rerun `main.py` with the new file.

## Interpreting The Plots

- `makespan vs repetition level`: lower is better; shows total finishing time.
- `average completion time vs repetition level`: lower is better; captures end-to-end latency from arrival to finish.
- `cache hit rate vs repetition level`: higher is better; shows how often routing benefits from warmed caches.
- `max server load vs repetition level`: lower is better; indicates how concentrated assignments are.
- `hybrid performance vs p`: shows the balance point between pure random load balancing and pure deterministic locality.

In general, you should expect:

- `power_of_two` to balance load well
- `hash_load_aware` to improve cache hits, especially under repeated task types
- `hybrid` to interpolate between the two behaviors

## Notes

- Stable hashing uses `hashlib.sha256`, not Python's built-in `hash`, so server rankings remain reproducible across runs.
- The project uses only the Python standard library plus `numpy`, `pandas`, and `matplotlib`.
