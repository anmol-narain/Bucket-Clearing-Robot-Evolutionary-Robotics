import numpy as np
from test_arena2 import BucketArena, RobotSensors


def compute_episode_fitness(
    buckets_out: int,
    total_buckets: int,
    time_moving_bucket: float,
    time_touching_wall: float,
    time_outside_arena: float,
    robot_left_arena: bool,
    max_time: float = 60.0,
) -> float:
    """
    Compute the scalar fitness score for a single episode.

    Formula:
        F = 5.0 * (n / N) + 0.5 * (s / T) - 1.0 * (w / T) - penalty_outside

    Where:
        n : buckets pushed outside the arena
        N : total buckets spawned in the episode (normalises across N in [3, 7])
        s : seconds spent actively moving a bucket (shaping term, capped below 1 bucket)
        w : seconds spent touching a wall (discourages wall-hugging / getting stuck)
        T : episode time limit (default 60s)
        penalty_outside : heavy penalty if the robot itself exits the arena
    """
    if total_buckets <= 0:
        return 0.0

    # Primary objective: fraction of buckets cleared [0.0, 5.0]
    bucket_term = 5.0 * (buckets_out / total_buckets)

    # Shaping term: reward actively moving buckets [0.0, 0.5]
    # Weighted at 0.5 so farming 's' for 60s never beats clearing even 1 of 7 buckets (5/7 = 0.71)
    push_term = 0.5 * (time_moving_bucket / max_time)

    # Penalty for rubbing/crashing into walls [0.0, 1.0]
    wall_penalty = 1.0 * (time_touching_wall / max_time)

    # Penalty for leaving the arena (robot must stay inside)
    outside_penalty = 0.0
    if robot_left_arena or time_outside_arena > 0.0:
        outside_penalty = 2.0 + 3.0 * (time_outside_arena / max_time)

    return float(bucket_term + push_term - wall_penalty - outside_penalty)


def run_episode(ann, config: dict, seed: int) -> dict:
    """Run one headless episode for a given ANN controller and environment seed."""
    arena = BucketArena(config, seed=seed, robot_pose=None)
    sensors = RobotSensors(arena)

    done = False
    while not done:
        obs = sensors.read()               # 12-element normalized sensor vector
        v_left, v_right = ann.forward(obs) # ANN returns 2 wheel speeds in [-v_max, v_max]
        done = arena.step(float(v_left), float(v_right))

    fitness = compute_episode_fitness(
        buckets_out=arena.buckets_collected,
        total_buckets=arena.initial_bucket_count,
        time_moving_bucket=arena.time_moving_bucket,
        time_touching_wall=arena.time_touching_wall,
        time_outside_arena=arena.time_outside_arena,
        robot_left_arena=arena.robot_left_arena,
        max_time=arena.max_time,
    )

    return {
        "seed": seed,
        "fitness": fitness,
        "buckets_out": arena.buckets_collected,
        "total_buckets": arena.initial_bucket_count,
        "time_elapsed": arena.time_elapsed,
        "time_moving_bucket": arena.time_moving_bucket,
        "time_touching_wall": arena.time_touching_wall,
        "robot_left_arena": arena.robot_left_arena,
    }


def evaluate_individual(
    ann,
    config: dict,
    episode_seeds: list[int],
    aggregation: str = "mean",
) -> tuple[float, list[dict]]:
    """
    Test one robot (ANN) across multiple arena scenarios and return a single fitness score.

        Each integer in `episode_seeds` creates one specific arena scenario that is (1 episode)
        with its own bucket count, bucket positions, and robot start pose.

        For example,passing 3 seeds runs 3 different scenarios (for each generation) and combines their scores (exaple - average)
        so a robot cannot win just by getting one lucky layout.

    Arguments:
        ann - Controller instance implementing .forward(inputs) -> (v_left, v_right).
        config - Simulation config dictionary.
        episode_seeds - List of seeds (e.g., 3 to 5 episodes per generation).Pass the SAME seeds to every individual in a generation(Common Random Numbers) for fair comparison.
        aggregation - How to combine episode scores ('mean', 'median', or 'min').

    Returns:
        (aggregated_fitness, list_of_episode_metrics)
    """
    metrics = [run_episode(ann, config, seed) for seed in episode_seeds]
    scores = []
    for m in metrics:
        scores.append(m["fitness"])

    if aggregation == "mean":
        agg_score = float(np.mean(scores))
    elif aggregation == "median":
        agg_score = float(np.median(scores))
    elif aggregation == "min":
        agg_score = float(np.min(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return agg_score, metrics
