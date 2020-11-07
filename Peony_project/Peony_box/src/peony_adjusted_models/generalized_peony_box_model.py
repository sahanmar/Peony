import pymongo
import numpy as np
import torch

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
        acquisition_function: Optional[
            Callable[[np.ndarray, int, np.ndarray], np.ndarray]
        ],
        greedy_coef_decay: Optional[Callable[[int], float]],
        reset_after_adding_new_samples: bool = True,
        ascquisition_func_ratio: float = 1,
    ):
        self.model = model
        self.transformator = transformator
        self.active_learning_step = active_learning_step
        self.training_dataset: Dict[str, np.ndarray] = {}
        self.acquisition_function = acquisition_function
        self.epsilon_greedy_coef = 0.0
        self.active_learning_iteration = 0
        self.reset_after_adding_new_samples = reset_after_adding_new_samples
        self.ascquisition_func_ratio = (
            1 if active_learning_step == 1 else ascquisition_func_ratio
        )
        if greedy_coef_decay:
            self.greedy_coef_decay = greedy_coef_decay
        else:
            self.greedy_coef_decay = sigmoid_decay

    def fit(
        self,
        training_instances: Union[List[Dict[str, Any]], List[torch.Tensor]],
        training_labels: Union[List[Any], List[int]],
        transformation_needed: bool = True,
    ) -> Optional[List[Any]]:

        fit_output: List[Any] = []

        if transformation_needed:
            print("transforming instances for model training...")
            training_instances = self.transformator.transform_instances(
                training_instances
            )
            print("transforming labels for model training...")
            training_labels = self.transformator.transform_labels(training_labels)

        instances = easy_colate(training_instances)  # type: ignore
        labels = torch.tensor(training_labels, dtype=torch.int64)
        if self.training_dataset == {}:
            self.training_dataset["training_instances"] = instances
            self.training_dataset["training_labels"] = labels
        else:
            self.training_dataset["training_instances"] = self._concatenate(
                self.training_dataset["training_instances"], instances
            )
            self.training_dataset["training_labels"] = self._concatenate(
                self.training_dataset["training_labels"], labels
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
        instances: Union[List[Dict[str, Any]], List[torch.Tensor]],
        transformation_needed: bool = True,
    ) -> List[Any]:
        if transformation_needed:
            print("transforming instances for model prediction...")
            instances = self.transformator.transform_instances(instances)

        instances = easy_colate(instances)  # type: ignore
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
                active_learning_samples = int(
                    round(self.active_learning_step * self.ascquisition_func_ratio)
                )
                return np.concatenate(
                    (
                        self.acquisition_function(
                            np.asarray(predicted), active_learning_samples, instances
                        ),
                        random_sampling(
                            predicted,
                            self.active_learning_step - active_learning_samples,
                        ),
                    )
                )
        else:
            self.active_learning_iteration += self.active_learning_step
            return random_sampling(predicted, self.active_learning_step)

    def add_new_learning_samples(
        self,
        instances: Union[List[Dict[str, Any]], np.ndarray],
        labels: Union[List[Any], np.ndarray],
        transformation_needed: bool = True,
    ) -> None:
        if self.reset_after_adding_new_samples:
            self.reset()
        if transformation_needed:
            self.fit(instances, labels)
        else:
            self.fit(instances, labels, transformation_needed=False)

    @staticmethod
    def _concatenate(
        first: Union[torch.Tensor, csc_matrix], second: Union[torch.Tensor, csc_matrix]
    ) -> Union[torch.Tensor, csc_matrix]:
        try:
            concatenated = torch.cat((first, second), dim=0)
        except:
            concatenated = vstack([first, second])
        return concatenated

    @staticmethod
    def _one_hot_encoding(instances: np.ndarray) -> np.ndarray:
        ...
        # onehot_encoder = OneHotEncoder(sparse=False)
        # return [onehot_encoder.fit_transform(integer_encoded) for ]


def easy_colate(embeddings: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(embeddings, dim=0)
