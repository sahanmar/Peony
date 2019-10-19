import numpy as np

from sklearn.ensemble import forest
from sklearn.base import clone
from typing import List


NUM_ENSEMBLES = 10


class PeonyRandomForest:
    def __init__(self, rand_sample_ratio: int):
        self.ensembles = [
            forest.RandomForestClassifier(n_estimators=20) for i in range(NUM_ENSEMBLES)
        ]
        self.num_ensembles = NUM_ENSEMBLES
        self.num_of_samples = 0
        self.rand_sample_ratio = rand_sample_ratio

    def fit(self, instances: np.ndarray, labels: np.ndarray):
        self.num_of_samples = int(instances.shape[0] * self.rand_sample_ratio)
        for index in range(self.num_ensembles):
            indices = np.random.choice(
                instances.shape[0], self.num_of_samples, replace=False
            )
            self.ensembles[index].fit(instances[indices, :], labels[indices])

    def predict(self, instances: np.ndarray) -> List[np.ndarray]:
        predicted = [
            self.ensembles[index].predict(instances)
            for index in range(self.num_ensembles)
        ]
        return predicted

    def reset(self) -> None:
        self.ensembles = [
            clone(self.ensembles[index]) for index in range(self.num_ensembles)
        ]
