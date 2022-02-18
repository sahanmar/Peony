import torch

import numpy as np

from typing import Optional, List
from omegaconf import base
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering

from itertools import combinations, product, count

from tqdm import tqdm

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


def batch_bald(
    labels: np.ndarray,
    len_rand_samples: int,
    instances: np.ndarray,
    randomly_sampled_combinations: int = 1000,
    aggregate_embeddings: bool = True,
) -> np.ndarray:

    # if aggregate_embeddings:
    #     instances = torch.stack([torch.mean(torch.stack(row, dim=0), dim=0) for row in instances], dim=0)
    # clustering = AgglomerativeClustering(linkage="average").fit(instances)

    nn_dist_sample_num = len(labels)
    idx_combinations = [
        np.random.randint(len(instances), size=len_rand_samples).astype(int)
        for _ in range(randomly_sampled_combinations)
    ]

    max_mutual_information = -np.inf
    idx_comb_2_return = None

    mutual_information_dist = []

    for idx_comb in tqdm(idx_combinations):
        comb_labels = labels[:, idx_comb]

        expected_class_entropy = np.sum(entropy(comb_labels, base=BASE, axis=2)) / nn_dist_sample_num
        mutual_labels_prob_dist_entropy = entropy(
            [
                np.sum(
                    [
                        np.prod([sample[i] for sample, i in zip(nn_dist_sample, cartesian_comb)])
                        for nn_dist_sample in comb_labels
                    ],
                    axis=0,
                )
                / nn_dist_sample_num
                for cartesian_comb in cartesian_ids_product_sampling(max_int=2, size=len(comb_labels[0]))
            ],
            base=BASE,
        )

        mutual_information = mutual_labels_prob_dist_entropy - expected_class_entropy

        mutual_information_dist.append(mutual_information)

        if mutual_information > max_mutual_information:

            max_mutual_information = mutual_information
            idx_comb_2_return = idx_comb

    import IPython

    IPython.embed()

    return np.asarray(idx_comb_2_return)


def cartesian_ids_product_sampling(
    max_int: int, size: int, randomly_sampled_combinations: int = 1000
) -> List[List[int]]:
    return [np.random.randint(max_int, size=size).astype(int) for _ in range(randomly_sampled_combinations)]
