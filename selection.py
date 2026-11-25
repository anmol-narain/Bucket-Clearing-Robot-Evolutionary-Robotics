import numpy as np
from crossover import crossover
from ann import ANN

def selection(
    fitness: np.ndarray, population: list[ANN], mu: int, lambda_: int
) -> list[np.ndarray]:
    """Return the best mu genomes followed by lambda_ crossover children."""
    if fitness.shape != (len(population),):
        raise ValueError("fitness must contain one score per individual")

    ranked = np.argsort(fitness, kind="stable")[::-1][:mu]
    survivors = [population[index] for index in ranked]
    weights = np.arange(mu, 0, -1, dtype=float)
    parent_pairs = np.random.choice(mu, size=(lambda_, 2), p=weights / weights.sum())
    children_weights = [crossover(survivors[first].weights, survivors[second].weights) for first, second in parent_pairs]
    children = list()
    for i in range(lambda_):
        children.append(ANN())
        children[i].weights = children_weights[i]
    return survivors + children