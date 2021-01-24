import numpy as np

from typing import Optional, List
from scipy.stats import entropy

BASE = None


def random_sampling(labels: np.ndarray, len_rand_samples: int) -> np.ndarray:
    labels = np.mean(labels, axis=0)
    return np.random.randint(labels.shape[0], size=len_rand_samples).astype(int)


def entropy_sampling(
    labels: np.ndarray, len_rand_samples: int, insatnces: np.ndarray
) -> np.ndarray:
    # insatnces are not used. They are given here in order to preserve unification
    max_entropy_indices: List[int] = []
    counts = [
        np.unique(labels[:, index], return_counts=True)[1]
        for index in range(len(labels[0, :]))
    ]
    prediction_entropy = np.asarray([entropy(dist, base=BASE) for dist in counts])
    for index in range(len_rand_samples):
        max_entropy_indices.append(np.argmax(prediction_entropy))
        prediction_entropy[max_entropy_indices[index]] = 0
    return np.asarray(max_entropy_indices)
