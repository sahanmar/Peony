from telnetlib import IP
from tkinter.tix import Tree
from matplotlib.pyplot import axis
import torch

import numpy as np

from typing import Dict, Any, List, Callable
from omegaconf import base
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering

from itertools import combinations, product, count

from tqdm import tqdm

BASE = 2
GUMBEL_BETA = 8


def random_sampling(labels: np.ndarray, batch_size: int) -> np.ndarray:
    labels = [np.argmax(l, axis=1) for l in labels]
    labels = np.mean(labels, axis=0)
    return np.random.randint(labels.shape[0], size=batch_size).astype(int)


def entropy_sampling(labels: np.ndarray, batch_size: int, instances: np.ndarray) -> np.ndarray:
    # instances are not used. They are given here in order to preserve unification

    dist_samples = len(labels)
    prediction_entropy = np.sum(entropy(labels, base=BASE, axis=2), axis=0) / dist_samples
    max_entropy_indices: List[int] = []

    for _ in range(batch_size):
        max_entropy_index = np.argmax(prediction_entropy)
        max_entropy_indices.append(max_entropy_index)
        prediction_entropy[max_entropy_index] = -np.inf

    return np.asarray(max_entropy_indices)


def batch_bald(labels: np.ndarray, batch_size: int, instances: np.ndarray) -> np.ndarray:

    nn_dist_sample_num = len(labels)
    idx_comb_2_return = []
    indices_to_iterate = list(range(len(labels[0])))

    print("Batch Bald sampling...")
    for values_sampled in tqdm(range(1, batch_size + 1)):
        max_mutual_information = -np.inf
        for idx in indices_to_iterate:
            if idx in idx_comb_2_return:
                continue
            idx_comb = idx_comb_2_return.copy()
            idx_comb.append(idx)

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
                    for cartesian_comb in product(*[[0, 1] for i in range(values_sampled)])
                ],
                base=BASE,
                axis=0,
            )

            mutual_information = mutual_labels_prob_dist_entropy - expected_class_entropy

            if mutual_information > max_mutual_information:

                max_mutual_information = mutual_information
                max_inform_idx = idx

        idx_comb_2_return.append(max_inform_idx)

    return np.asarray(idx_comb_2_return)


def hac_sampling(
    labels: np.ndarray,
    batch_size: int,
    instances: np.ndarray,
    acq_func: Callable[[Any], np.ndarray],
    aggregate_sentence_embeds=True,
    criterion: str = "size",
) -> np.ndarray:

    high_entropy_values = acq_func(labels, batch_size=10 * batch_size, instances=instances)

    if aggregate_sentence_embeds:
        instances = torch.stack([torch.mean(torch.stack(row, dim=0), dim=0) for row in instances], dim=0)

    model = AgglomerativeClustering(linkage="average").fit(instances.cpu())
    cluster_tree = dict(enumerate(model.children_, model.n_leaves_))

    cluster_criterion = []
    high_entropy_clusters = {}
    for high_entropy_val_idx in high_entropy_values:
        for cluster, components in cluster_tree.items():
            if high_entropy_val_idx in components:
                high_entropy_clusters.setdefault(cluster, []).append(high_entropy_val_idx)
                if cluster not in cluster_criterion:
                    cluster_criterion.append(cluster)

    if criterion == "size":
        cluster_criterion = [
            cluster
            for cluster, size in sorted(
                [(key, _get_cluster_sizes(key, cluster_tree)) for key in cluster_tree], key=lambda x: x[1]
            )
        ]

    ids_2_return = []
    for cluster_id in cluster_criterion:
        if len(ids_2_return) > batch_size:
            break
        if cluster_id in high_entropy_clusters and high_entropy_clusters[cluster_id]:
            index_to_add = high_entropy_clusters[cluster_id].pop(0)
            ids_2_return.append(index_to_add)

    if len(ids_2_return) > batch_size:
        print(f"Smth went wrong, only {len(ids_2_return)} values sampled instead of {batch_size}")

    return ids_2_return


def hac_entropy_sampling(
    labels: np.ndarray,
    batch_size: int,
    instances: np.ndarray,
    aggregate_sentence_embeds=True,
    criterion: str = "size",
) -> np.ndarray:

    return hac_sampling(labels, batch_size, instances, entropy_sampling, aggregate_sentence_embeds, criterion)


def bald(labels: np.ndarray) -> np.ndarray:
    nn_dist_sample_num = len(labels)
    expected_class_entropy = np.sum(entropy(labels, base=BASE, axis=2), axis=0) / nn_dist_sample_num
    mutual_labels_prob_dist_entropy = entropy(
        [
            np.sum(labels[:, :, 0], axis=0) / nn_dist_sample_num,
            np.sum(labels[:, :, 1], axis=0) / nn_dist_sample_num,
        ],
        base=BASE,
        axis=0,
    )

    return mutual_labels_prob_dist_entropy - expected_class_entropy


def bald_sampling(labels: np.ndarray, batch_size: int, instances: np.ndarray) -> np.ndarray:
    # instances are not used. They are given here in order to preserve unification

    prediction_bald = bald(labels)
    max_bald_indices: List[int] = []

    for _ in range(batch_size):
        max_entropy_index = np.argmax(prediction_bald)
        max_bald_indices.append(max_entropy_index)
        prediction_bald[max_entropy_index] = -np.inf

    return np.asarray(max_bald_indices)


def hac_bald_sampling(
    labels: np.ndarray,
    batch_size: int,
    instances: np.ndarray,
    aggregate_sentence_embeds=True,
    criterion: str = "size",
) -> np.ndarray:

    return hac_sampling(labels, batch_size, instances, bald_sampling, aggregate_sentence_embeds, criterion)


def power_bald(
    labels: np.ndarray,
    batch_size: int,
    instances: np.ndarray,
) -> np.ndarray:
    max_power_bald_indices: List[int] = []

    num_labels = len(labels[0])

    bald_score = (
        np.log(
            bald(labels),
        )
        + np.random.gumbel(scale=1 / GUMBEL_BETA, size=num_labels)
    )

    for _ in range(batch_size):
        max_entropy_index = np.argmax(bald_score)
        max_power_bald_indices.append(max_entropy_index)
        bald_score[max_entropy_index] = -np.inf

    return np.asarray(max_power_bald_indices)


def _get_cluster_sizes(key: int, cluster_tree: Dict[int, np.ndarray]) -> int:
    cluster_size = 0
    for leaf in cluster_tree[key]:
        if leaf in cluster_tree:
            cluster_size += _get_cluster_sizes(leaf, cluster_tree)
        else:
            cluster_size += 1

    return cluster_size
