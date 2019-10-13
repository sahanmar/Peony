import numpy as np

from sklearn.ensemble import forest
from sklearn.base import clone
from typing import List

NUM_ENSEMBLES = 10


class PeonyRandomForest:
    def __init__(self):
        self.ensembles = [
            forest.RandomForestClassifier(n_estimators=20) for i in range(NUM_ENSEMBLES)
        ]
        self.num_ensembles = NUM_ENSEMBLES

    def fit(self, instances: np.ndarray, labels: np.ndarray):
        [
            self.ensembles[index].fit(instances, labels)
            for index in range(self.num_ensembles)
        ]

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
