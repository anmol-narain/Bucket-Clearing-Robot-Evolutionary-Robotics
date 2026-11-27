import numpy as np
from crossover import crossover
from ann import ANN

def selection(
    fitness: np.ndarray, population: list[ANN], mu: int, lambda_: int
) -> list[np.ndarray]:
    """Return the best mu genomes followed by lambda_ crossover children."""
    if fitness.shape != (len(population),):
        raise ValueError("fitness must contain one score per individual")

    # 1. Survivor selection: Here we sort the best-to-worst and keep the top "mu" individuals
    ranked = np.argsort(fitness, kind="stable")[::-1][:mu]
    survivors = [population[index] for index in ranked]
    # 2. Rank-based parent probabilities (best survivor gets weight "mu", worst gets 1)
    rank_weights = np.arange(mu, 0, -1, dtype=float)
    probs = rank_weights / rank_weights.sum()

    # 3. Sample lambda_ pairs of DISTINCT parents (replace=False prevents self-crossover)
    parent_pairs = [
        np.random.choice(mu, size=2, replace=False, p=probs)
        for _ in range(lambda_)
    ]
    # parent_pairs = np.random.choice(mu, size=(lambda_, 2), p=weights / weights.sum())

    # 4. Create children by crossing over each parent pair's weights
    children = []
    for first, second in parent_pairs:
        child = ANN()
        child.weights = crossover(survivors[first].weights, survivors[second].weights)
        children.append(child)

    # Elites come first so main.py can mutate only new_generation[mu:]
    return survivors + children
