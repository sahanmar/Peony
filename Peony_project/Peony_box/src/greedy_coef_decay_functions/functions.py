import numpy as np


def sigmoid_decay(value: float) -> float:
    return np.exp(value - 50) / (np.exp(value - 50) + 1)
