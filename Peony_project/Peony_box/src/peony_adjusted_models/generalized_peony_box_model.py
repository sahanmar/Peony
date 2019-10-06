import pymongo
import numpy as np

from scipy.sparse import csc_matrix
from typing import Callable, Any, List, Dict, Optional, Union
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from scipy.sparse import vstack

# model: PeonyAdjustedModel
# TODO add default acquisition_function (Change optional to default
# acquisition_function and active_learning_step)
# TODO implement active_learning_sample and add_active_learning_sample


class GeneralizedPeonyBoxModel:
    def __init__(
        self,
        model: Any,
        transformator: Callable[[Union[List[Dict[str, Any]], List[str]]], np.ndarray],
        acquisition_function: Optional[Callable[[np.ndarray], List[int]]],
        active_learning_step: Optional[int],
    ):
        self.model = model
        self.transformator = transformator
        self.active_learning_step = active_learning_step
        self.training_dataset: Dict[str, np.ndarray] = {}

    def fit(
        self,
        training_instances: Union[List[Dict[str, Any]], np.ndarray],
        training_labels: Union[List[Any], np.ndarray],
        transformation_needed: bool = True,
    ) -> None:
        if transformation_needed:
            training_instances = self.transformator(training_instances)
            training_labels = self.transformator(training_labels)

        if self.training_dataset == {}:
            self.training_dataset["training_instances"] = training_instances
            self.training_dataset["training_labels"] = training_labels
        else:
            self.training_dataset["training_instances"] = self._concatenate(
                (self.training_dataset["training_instances"], training_instances)
            )
            self.training_dataset["training_labels"] = np.concatenate(
                (self.training_dataset["training_labels"], training_labels), axis=0
            )
        self.model.fit(
            self.training_dataset["training_instances"],
            self.training_dataset["training_labels"],
        )

    def predict(
        self,
        instances: Union[List[Dict[str, Any]], np.ndarray],
        transformation_needed: bool = True,
    ) -> List[Any]:
        if transformation_needed:
            instances = self.transformator(instances)
        return self.model.predict(instances)

    def reset(self) -> None:
        self.model.reset()

    def _concatenate(
        first: Union[np.ndarray, csc_matrix], second: Union[np.ndarray, csc_matrix]
    ) -> Union[np.ndarray, csc_matrix]:
        try:
            concatenated = np.concatenate((first, second), axis=0)
        except:
            concatenated = vstack([first, second])
        return concatenated

    def get_active_learning_sample(self):
        # TODO
        ...

    def add_active_learning_sample(self, annotated_label: np.ndarray):
        # TODO
        ...
