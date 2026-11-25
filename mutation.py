import numpy as np

def mutate(genome: np.ndarray, mutation_rate: float = 0.2, std: float = 0.01) -> np.ndarray:
    # mutation function
    # genome is the matrix of weights, mutation_rate is proporiton of the matrix entries mutated,
    # std is the standard deviation, mean is the mean of the noise distribution

    mutation_mask = np.random.random(genome.shape) < mutation_rate
    noise = np.random.normal(loc=0.0, scale=std, size=genome.shape)
    return genome + np.where(mutation_mask, noise, 0)
