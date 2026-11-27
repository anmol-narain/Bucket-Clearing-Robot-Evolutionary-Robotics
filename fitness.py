import numpy as np
from test_arena2 import BucketArena, RobotSensors


def _dist_to_nearest_gap(pos, gap_centers: list[np.ndarray]) -> float:
    """Return Euclidean distance from a 2D Box2D position to the closest wall gap."""
    p = np.array([pos[0], pos[1]], dtype=float)
    return min(float(np.linalg.norm(p - g)) for g in gap_centers)

def compute_episode_fitness(
    buckets_out: int,
    total_buckets: int,
    time_moving_bucket: float,
    time_touching_wall: float,
    time_outside_arena: float,
    robot_left_arena: bool,
    max_time: float = 60.0,
    *,
    time_idle: float = 0.0,
    bucket_approach: float = 0.0,
    bucket_progress: float = 0.0,
) -> float:
    """
    Compute the scalar fitness score for a single episode.

    Formula:
        F = 5.0 * (n / N) + 0.3 * (s / T) + 0.3 * a + p
            - (w / T) - (i / T) - penalty_outside

    Where:
        n : buckets pushed outside the arena
        N : total buckets spawned in the episode (normalises across N in [3, 7])
        p : mean normalized progress of all buckets toward their nearest gap, in [0, 1]
        s : seconds spent actively moving a bucket (shaping term, capped below 1 bucket)
        w : seconds spent touching a wall (discourages wall-hugging / getting stuck)
        i : seconds spent nearly stationary
        a : normalized improvement in closest approach to a bucket, in [0, 1]
        T : episode time limit (default 60s)
        penalty_outside : heavy penalty if the robot itself exits the arena
    """
    # Primary objective: fraction of buckets cleared [0.0, 5.0]
    bucket_term = 5.0 * (buckets_out / total_buckets) if total_buckets > 0 else 0.0

    # Directional shaping: reward pushing buckets closer to any exit gap [0.0, 1.0]
    progress_term = 1.0 * bucket_progress

    # Shaping term: reward actively moving buckets [0.0, 0.5]
    # The pushing term alone is smaller than the reward for clearing one of seven buckets.
    push_term = 0.3 * (time_moving_bucket / max_time)
    approach_term = 0.3 * bucket_approach

    # Penalty for rubbing/crashing into walls [0.0, 1.0]
    wall_penalty = 1.0 * (time_touching_wall / max_time)
    idle_penalty = time_idle / max_time

    # Penalty for leaving the arena (robot must stay inside)
    outside_penalty = 0.0
    if robot_left_arena or time_outside_arena > 0.0:
        outside_penalty = 2.0 + 3.0 * (time_outside_arena / max_time)

    return float(bucket_term + push_term + progress_term + approach_term - wall_penalty - idle_penalty - outside_penalty)


def run_episode(ann, config: dict, seed: int) -> dict:
    """Run one headless episode for a given ANN controller and environment seed."""
    arena = BucketArena(config, seed=seed, robot_pose=None)
    sensors = RobotSensors(arena)

    # Gap centers on the 4 outer walls (bottom, top, left, right)
    w_arena = config["arena"]["width"]
    h_arena = config["arena"]["height"]
    gap_centers = [
        np.array([w_arena / 2.0, 0.0]),
        np.array([w_arena / 2.0, h_arena]),
        np.array([0.0, h_arena / 2.0]),
        np.array([w_arena, h_arena / 2.0]),
    ]

    # Record each bucket's initial distance to its nearest gap (keyed by bucket id)
    initial_gap_dists = {
        b.userData["id"]: _dist_to_nearest_gap(b.position, gap_centers)
        for b in arena.bucket_bodies
    }

    def nearest_bucket_distance():
        robot_position = arena.robot_body.position
        return min(
            ((bucket.position - robot_position).length for bucket in arena.bucket_bodies),
            default=float("inf"),
        )

    initial_distance = nearest_bucket_distance()
    closest_distance = initial_distance
    done = False
    while not done:
        obs = sensors.read()               # 12-element normalized sensor vector
        v_left, v_right = ann.forward(obs) # ANN returns 2 wheel speeds in [-v_max, v_max]
        done = arena.step(float(v_left), float(v_right))
        closest_distance = min(closest_distance, nearest_bucket_distance())

    # Score the episode's best approach once, so repeated passes cannot farm points.
    contact_distance = config["robot"]["radius"] + config["buckets"]["radius"]
    initial_gap = initial_distance - contact_distance
    bucket_approach = (
        float(np.clip((initial_distance - closest_distance) / initial_gap, 0.0, 1.0))
        if 0.0 < initial_gap < float("inf") else 0.0
    )

    # 2. Mean bucket-to-gap progress across all N buckets [0, 1]
    remaining_by_id = {b.userData["id"]: b for b in arena.bucket_bodies}
    progress_scores = []
    for b_id, init_d in initial_gap_dists.items():
        if b_id not in remaining_by_id:
            # Bucket was pushed out through a gap -> full progress
            progress_scores.append(1.0)
        elif init_d > 1e-6:
            final_d = _dist_to_nearest_gap(remaining_by_id[b_id].position, gap_centers)
            progress_scores.append(float(np.clip((init_d - final_d) / init_d, 0.0, 1.0)))
        else:
            progress_scores.append(0.0)
    bucket_progress = float(np.mean(progress_scores)) if progress_scores else 0.0

    fitness = compute_episode_fitness(
        buckets_out=arena.buckets_collected,
        total_buckets=arena.initial_bucket_count,
        time_moving_bucket=arena.time_moving_bucket,
        time_touching_wall=arena.time_touching_wall,
        time_outside_arena=arena.time_outside_arena,
        robot_left_arena=arena.robot_left_arena,
        max_time=arena.max_time,
        time_idle=arena.time_idle,
        bucket_approach=bucket_approach,
        bucket_progress=bucket_progress,
    )

    return {
        "seed": seed,
        "fitness": fitness,
        "buckets_out": arena.buckets_collected,
        "total_buckets": arena.initial_bucket_count,
        "time_elapsed": arena.time_elapsed,
        "time_moving_bucket": arena.time_moving_bucket,
        "time_touching_wall": arena.time_touching_wall,
        "time_idle": arena.time_idle,
        "bucket_approach": bucket_approach,
        "bucket_progress": bucket_progress,
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
