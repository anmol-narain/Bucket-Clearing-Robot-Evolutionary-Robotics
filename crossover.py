import numpy as np


def crossover(ind1: np.ndarray, ind2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    BLX-alpha (blend) crossover for real-valued weight matrices.
    Samples each gene randomly in [-alpha, 1 + alpha] between ind1 and ind2,
    preserving population variance instead of shrinking weights toward the midpoint.
    """
    t = np.random.uniform(-alpha, 1.0 + alpha, size=ind1.shape)
    return t * ind1 + (1.0 - t) * ind2
