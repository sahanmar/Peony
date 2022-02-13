from telnetlib import IP
import pymongo
import numpy as np
import torch

from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Callable, Any, List, Dict, Optional, Union, Tuple

from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from Peony_box.src.transformators.generalized_transformator import Transformator
from Peony_box.src.acquisition_functions.functions import random_sampling
from Peony_box.src.greedy_coef_decay_functions.functions import sigmoid_decay


BATCH_SIZE = 32


def easy_colate(inputs) -> Tuple[torch.Tensor, torch.Tensor]:

    embeddings, labels = zip(*inputs)

    return (
        torch.stack([torch.mean(torch.stack(row, dim=0), dim=0) for row in embeddings], dim=0),
        torch.tensor(labels, dtype=torch.int64),
    )


def lstm_colate(inputs) -> List[Any]:

    embeddings, target = zip(*inputs)
    zipped = zip(
        [torch.stack(sentence, dim=0) for sentence in embeddings],
        target,
        list(range(len(embeddings))),
        [len(sentence) for sentence in embeddings],
    )

    embeddings, labels, indices, seq_lengths = zip(*sorted(zipped, key=lambda x: x[-1], reverse=True))

    indices = sorted(range(len(indices)), key=lambda k: indices[k])

    return [
        (
            pad_sequence(embeddings, batch_first=True),
            torch.tensor(seq_lengths, dtype=torch.long),
            torch.tensor(indices, dtype=torch.long),
        ),
        torch.tensor(labels, dtype=torch.int64),
    ]


class PeonyDataset(Dataset):
    def __init__(self, instances, labels):
        "Initialization"

        self.instances, self.labels = instances, labels
        self.features_size = self.instances[0][0].size()[0] if self.instances else 0

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.instances)

    def __getitem__(self, index):
        "Generates one sample of data"

        return self.instances[index], self.labels[index]


class GeneralizedPeonyBoxModel:
    def __init__(  # type: ignore
        self,
        model: Any,
        transformator: Transformator,
        active_learning_step: int,
        acquisition_function: Optional[Callable[[np.ndarray, int, np.ndarray], np.ndarray]],
        greedy_coef_decay: Optional[Callable[[int], float]],
        reset_after_adding_new_samples: bool = True,
        ascquisition_func_ratio: float = 1,
        collate: Callable[[List[Any]], Any] = easy_colate,
    ):
        self.model = model
        self.transformator = transformator
        self.active_learning_step = active_learning_step
        self.training_dataset: Dict[str, np.ndarray] = {}
        self.acquisition_function = acquisition_function
        self.epsilon_greedy_coef = 0.0
        self.active_learning_iteration = 0
        self.reset_after_adding_new_samples = reset_after_adding_new_samples
        self.ascquisition_func_ratio = 1 if active_learning_step == 1 else ascquisition_func_ratio
        self.collate = collate
        if greedy_coef_decay:
            self.greedy_coef_decay = greedy_coef_decay
        else:
            self.greedy_coef_decay = sigmoid_decay

    def fit(
        self,
        instances: Union[List[Dict[str, Any]], List[List[torch.Tensor]]],
        labels: Union[List[Any], List[int]],
        transformation_needed: bool = True,
    ) -> Optional[List[Any]]:

        fit_output: List[Any] = []

        if transformation_needed:
            print("transforming instances for model training...")
            instances = self.transformator.transform_instances(instances)
            print("transforming labels for model training...")
            labels = self.transformator.transform_labels(labels)

        # labels = torch.tensor(labels, dtype=torch.int64)
        # instances = easy_colate(training_instances)  # type: ignore
        if self.training_dataset == {}:
            self.training_dataset["training_instances"] = instances
            self.training_dataset["training_labels"] = labels
        else:
            self.training_dataset["training_instances"] = (
                self.training_dataset["training_instances"] + instances
            )
            self.training_dataset["training_labels"] = self.training_dataset["training_labels"] + labels

        training_dataloader = PeonyDataset(
            self.training_dataset["training_instances"],
            self.training_dataset["training_labels"],
        )

        fit_output.append(
            self.model.fit(
                DataLoader(
                    training_dataloader,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    collate_fn=self.collate,
                ),
                training_dataloader.features_size,
            )
        )

        if None not in fit_output:
            return fit_output
        else:
            return None

    def predict(
        self,
        instances: Union[List[Dict[str, Any]], List[List[torch.Tensor]]],
        transformation_needed: bool = True,
    ) -> List[Any]:
        if transformation_needed:
            print("transforming instances for model prediction...")
            instances = self.transformator.transform_instances(instances)

        pred_dataset = PeonyDataset(
            instances,
            torch.zeros((len(instances))),
        )

        predicted = self.model.predict(
            DataLoader(
                pred_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=self.collate,
            )
        )

        import IPython

        IPython.embed()

        return np.mean([np.argmax(pred, axis=1) for pred in predicted], axis=0)

    def reset(self) -> None:
        self.model.reset()

    def get_learning_samples(
        self,
        instances: Union[List[Dict[str, Any]], List[torch.Tensor]],
        transformation_needed: bool = True,
    ) -> np.ndarray:
        if transformation_needed:
            print("transforming instances for model getting learning sample...")
            instances = self.transformator.transform_instances(instances)

        pred_dataset = PeonyDataset(
            instances,
            torch.zeros((len(instances))),
        )
        predicted = self.model.predict(
            DataLoader(
                pred_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=self.collate,
            )
        )
        if self.acquisition_function is not None:
            if np.random.uniform(0, 1) > self.epsilon_greedy_coef:
                self.epsilon_greedy_coef = self.greedy_coef_decay(self.active_learning_iteration)
                self.active_learning_iteration += self.active_learning_step
                return random_sampling(predicted, self.active_learning_step)
            else:
                self.epsilon_greedy_coef = self.greedy_coef_decay(self.active_learning_iteration)
                self.active_learning_iteration += self.active_learning_step
                active_learning_samples = int(round(self.active_learning_step * self.ascquisition_func_ratio))
                return np.concatenate(
                    (
                        self.acquisition_function(np.asarray(predicted), active_learning_samples, instances),
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
