import numpy as np

def crossover(ind1: np.ndarray, ind2: np.ndarray) -> np.ndarray:
    # takes the genom of two individuals and return the arithmetic mean
    # the size of the genoms is assumed to match
    return (ind1 + ind2)/2
