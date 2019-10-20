import numpy as np


def sigmoid_decay(value: float) -> float:
    return np.exp(value - 70) / (np.exp(value - 70) + 1)
