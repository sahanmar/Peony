import numpy as np

from sklearn.ensemble import forest
from sklearn.base import clone


class PeonyRandomForest:
    def __init__(self):
        self.clf = forest.RandomForestClassifier(n_estimators=100)

    def fit(self, instances: np.ndarray, labels: np.ndarray):
        self.clf.fit(instances, labels)

    def predict(self, instances: np.ndarray) -> np.ndarray:
        return self.clf.predict(instances)

    def reset(self) -> None:
        self.clf = clone(self.clf)
