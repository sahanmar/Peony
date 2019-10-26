import pymongo
import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from sklearn.preprocessing import OneHotEncoder

from typing import Callable, Any, List, Dict, Optional, Union

from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from Peony_box.src.transformators.generalized_transformator import Transformator
from Peony_box.src.acquisition_functions.functions import random_sampling
from Peony_box.src.greedy_coef_decay_functions.functions import sigmoid_decay


class GeneralizedPeonyBoxModel:
    def __init__(
        self,
        model: Any,
        transformator: Transformator,
        active_learning_step: int,
        acquisition_function: Optional[Callable[[np.ndarray, int], np.ndarray]],
        greedy_coef_decay: Optional[Callable[[int], float]],
    ):
        self.model = model
        self.transformator = transformator
        self.active_learning_step = active_learning_step
        self.training_dataset: Dict[str, np.ndarray] = {}
        self.acquisition_function = acquisition_function
        self.epsilon_greedy_coef = 0.0
        self.active_learning_iteration = 0
        if greedy_coef_decay:
            self.greedy_coef_decay = greedy_coef_decay
        else:
            self.greedy_coef_decay = sigmoid_decay

    def fit(
        self,
        training_instances: Union[List[Dict[str, Any]], np.ndarray],
        training_labels: Union[List[Any], np.ndarray],
        transformation_needed: bool = True,
    ) -> Optional[List[Any]]:

        fit_output: List[Any] = []

        if transformation_needed:
            # self.transformator.reset()
            print("transforming instances for model training...")
            training_instances = self.transformator.transform_instances(
                training_instances
            )
            print("transforming labels for model training...")
            training_labels = self.transformator.transform_labels(training_labels)

        if self.training_dataset == {}:
            self.training_dataset["training_instances"] = training_instances
            self.training_dataset["training_labels"] = training_labels
        else:
            self.training_dataset["training_instances"] = self._concatenate(
                self.training_dataset["training_instances"], training_instances
            )
            self.training_dataset["training_labels"] = np.concatenate(
                (self.training_dataset["training_labels"], training_labels), axis=0
            )
        fit_output.append(
            self.model.fit(
                self.training_dataset["training_instances"],
                self.training_dataset["training_labels"],
            )
        )

        if None not in fit_output:
            return fit_output
        else:
            return None

    def predict(
        self,
        instances: Union[List[Dict[str, Any]], np.ndarray],
        transformation_needed: bool = True,
    ) -> List[Any]:
        if transformation_needed:
            print("transforming instances for model prediction...")
            instances = self.transformator.transform_instances(instances)
        predicted = self.model.predict(instances)
        return np.mean(predicted, axis=0)

    def reset(self) -> None:
        self.model.reset()

    def get_learning_samples(
        self,
        instances: Union[List[Dict[str, Any]], np.ndarray],
        transformation_needed: bool = True,
    ) -> np.ndarray:
        if transformation_needed:
            print("transforming instances for model getting learning sample...")
            instances = self.transformator.transform_instances(instances)
        predicted = self.model.predict(instances)
        if self.acquisition_function is not None:
            if np.random.uniform(0, 1) > self.epsilon_greedy_coef:
                self.epsilon_greedy_coef = self.greedy_coef_decay(
                    self.active_learning_iteration
                )
                self.active_learning_iteration += self.active_learning_step
                return random_sampling(predicted, self.active_learning_step)
            else:
                self.epsilon_greedy_coef = self.greedy_coef_decay(
                    self.active_learning_iteration
                )
                self.active_learning_iteration += self.active_learning_step
                return self.acquisition_function(predicted, self.active_learning_step)
        else:
            self.active_learning_iteration += self.active_learning_step
            return random_sampling(predicted, self.active_learning_step)

    def add_new_learning_samples(
        self,
        instances: Union[List[Dict[str, Any]], np.ndarray],
        labels: Union[List[Any], np.ndarray],
    ) -> None:
        self.reset()
        if isinstance(instances, list):
            self.fit(instances, labels)
        else:
            self.fit(instances, labels, transformation_needed=False)

    @staticmethod
    def _concatenate(
        first: Union[np.ndarray, csc_matrix], second: Union[np.ndarray, csc_matrix]
    ) -> Union[np.ndarray, csc_matrix]:
        try:
            concatenated = np.concatenate((first, second), axis=0)
        except:
            concatenated = vstack([first, second])
        return concatenated

    @staticmethod
    def _one_hot_encoding(instances: np.ndarray) -> np.ndarray:
        ...
        # onehot_encoder = OneHotEncoder(sparse=False)
        # return [onehot_encoder.fit_transform(integer_encoded) for ]
