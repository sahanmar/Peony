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


def false_positive_sampling(
    labels: np.ndarray, len_rand_samples: int, insatnces: np.ndarray
) -> np.ndarray:
    highest_entropy_instance = insatnces[
        entropy_sampling(labels, len_rand_samples, insatnces), :
    ]
    dist = [
        np.linalg.norm(highest_entropy_instance[0] - insatnces[index])
        for index in range(len(insatnces))
    ]
    dist_indices = sorted(range(len(insatnces)), key=lambda k: dist[k])
    false_positive_indices: List[int] = []
    prediction_entropy = entropy(labels, base=BASE)  # bad naming for variable
    for index in range(len(labels[0])):
        false_positive_indices.append(np.argmax(prediction_entropy))
        prediction_entropy[false_positive_indices[index]] = 0
    closest_false_positive_indices: List[int] = []
    for index in false_positive_indices:
        if (
            index in dist_indices[: int(round(0.2 * len(dist_indices)))]
            and index not in closest_false_positive_indices
        ):
            closest_false_positive_indices.append(index)
        if len(closest_false_positive_indices) == len_rand_samples:
            break
    if len(closest_false_positive_indices) < len_rand_samples:
        for index in highest_entropy_instance:
            if index not in closest_false_positive_indices:
                closest_false_positive_indices.append(index)
            if len(closest_false_positive_indices) == len_rand_samples:
                break
    return np.asarray(closest_false_positive_indices)
