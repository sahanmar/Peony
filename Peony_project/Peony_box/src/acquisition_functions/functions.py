from telnetlib import IP
import numpy as np

from typing import Optional, List
from omegaconf import base
from scipy.stats import entropy

from itertools import combinations, product

BASE = 2


def random_sampling(labels: np.ndarray, len_rand_samples: int) -> np.ndarray:
    labels = [np.argmax(l, axis=1) for l in labels]
    labels = np.mean(labels, axis=0)
    return np.random.randint(labels.shape[0], size=len_rand_samples).astype(int)


def entropy_sampling(labels: np.ndarray, len_rand_samples: int, insatnces: np.ndarray) -> np.ndarray:
    # insatnces are not used. They are given here in order to preserve unification
    labels = [np.argmax(l, axis=1) for l in labels]
    max_entropy_indices: List[int] = []
    counts = [np.unique(labels[:, index], return_counts=True)[1] for index in range(len(labels[0, :]))]
    prediction_entropy = np.asarray([entropy(dist, base=BASE) for dist in counts])
    for index in range(len_rand_samples):
        max_entropy_indices.append(np.argmax(prediction_entropy))
        prediction_entropy[max_entropy_indices[index]] = 0
    return np.asarray(max_entropy_indices)


def batch_bald(labels: np.ndarray, len_rand_samples: int, insatnces: np.ndarray) -> np.ndarray:

    import IPython

    IPython.embed()

    nn_dist_sample_num = len(labels)
    idx_combinations = combinations(range(len(insatnces)), len_rand_samples)

    min_mutual_information = np.inf
    idx_comb_2_return = None

    for idx_comb in idx_combinations:
        comb_labels = labels[:, idx_comb]

        import IPython

        IPython.embed()

        expected_class_entropy = np.sum(entropy(comb_labels, base=BASE, axis=2)) / nn_dist_sample_num
        mutual_labels_prob_dist_entropy = entropy(
            np.sum(
                [
                    [np.prod(labels_dist_sample) for labels_dist_sample in product(*nn_dist_sample)]
                    for nn_dist_sample in labels
                ],
                axis=0,
            )
            / nn_dist_sample_num,
            base=BASE,
        )

        import IPython

        IPython.embed()

        mutual_information = mutual_labels_prob_dist_entropy - expected_class_entropy
        if mutual_information < min_mutual_information:

            import IPython

            IPython.embed()

            min_mutual_information = mutual_information
            idx_comb_2_return = comb_labels

    return np.asarray(idx_comb_2_return)
