import numpy as np


def sigmoid_decay(value: int) -> float:
    return np.exp((value - 40) * 0.1) / (np.exp((value - 40) * 0.1) + 1)
