import numpy as np

from typing import Optional, List
from scipy.stats import entropy

BASE = None


def random_sampling(instaces: np.ndarray, len_rand_samples: int) -> np.ndarray:
    instaces = np.mean(instaces, axis=0)
    return np.random.randint(instaces.shape[0], size=len_rand_samples).astype(int)


def entropy_sampling(instaces: np.ndarray, len_rand_samples: int) -> np.ndarray:
    max_entropy_indices: List[int] = []
    prediction_entropy = entropy(instaces, base=BASE)
    for index in range(len_rand_samples):
        max_entropy_indices.append(np.argmax(prediction_entropy))
        prediction_entropy[max_entropy_indices[index]] = 0
    return np.asarray(max_entropy_indices)
