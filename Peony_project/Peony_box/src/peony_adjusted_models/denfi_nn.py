import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

from Peony_box.src.peony_adjusted_models.neural_nets_architecture import (
    NeuralNet,
    NeuralNetLSTM,
)

NUM_ENSEMBLES = 5
EPOCHS = 2500
HOT_START_EPOCHS = 700
WEIGHTS_VARIANCE = 0.3
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001

# NUM_ENSEMBLES = 1
# EPOCHS = 150
# HOT_START_EPOCHS = 40
# WEIGHTS_VARIANCE = 0.1
# # Device configuration
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LEARNING_RATE = 0.001

neural_network = NeuralNet


class PeonyDENFIFeedForwardNN:
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        rand_sample_ratio: int,
        cold_start: bool = True,
    ):

        self.num_ensembles = NUM_ENSEMBLES

        self.model: Optional[List[NeuralNet]] = None
        self.criterion: Optional[List[nn.CrossEntropyLoss]] = None
        self.optimizer: Optional[List[torch.optim.Adam]] = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False
        self.rand_sample_ratio = rand_sample_ratio
        self.variance = WEIGHTS_VARIANCE
        self.loss_sequence: List[List[float]] = []
        self.cold_start = cold_start
        self.hot_start_epochs = HOT_START_EPOCHS

    def fit(self, data: DataLoader, features_size: int) -> Optional[List[str]]:

        loss_list: List[str] = []

        self.loss_sequence = []

        if self.initialized is False:
            self.model = [
                neural_network(features_size, self.hidden_size, self.num_classes).to(DEVICE)
                for i in range(self.num_ensembles)
            ]
            self.criterion = [nn.CrossEntropyLoss() for i in range(self.num_ensembles)]
            self.optimizer = [
                # torch.optim.SGD(
                #     self.model[i].parameters(), lr=LEARNING_RATE, momentum=0.9
                # )
                torch.optim.Adam(self.model[i].parameters(), lr=LEARNING_RATE)
                for i in range(self.num_ensembles)
            ]
            for index in range(self.num_ensembles):
                self._init_normal_weights(index)

            self.initialized = True

        initial_loss_per_ensemble: List[str] = []
        fitted_loss_per_ensemble: List[str] = []

        for index in range(self.num_ensembles):

            loss_sequence_per_ensemble: List[float] = []

            if self.cold_start is False:
                with torch.no_grad():
                    for param in self.model[index].parameters():
                        param.add_(torch.randn(param.size()).to(DEVICE) * self.variance)

            for epoch in range(self.num_epochs):

                for instances, labels in data:
                    # Forward pass
                    outputs = self.model[index](instances)
                    loss = self.criterion[index](outputs, labels.to(DEVICE))
                    loss_sequence_per_ensemble.append(float(loss.cpu().detach().numpy()))
                    # Backward and optimize
                    self.optimizer[index].zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer[index].step()

                    if epoch == 0:
                        initial_loss_per_ensemble.append(str(loss.cpu().detach().numpy()))

            self.loss_sequence.append(loss_sequence_per_ensemble)
            fitted_loss_per_ensemble.append(str(loss.cpu().detach().numpy()))

        loss_list.append(f"starting losses are {'| '.join(initial_loss_per_ensemble)}")
        loss_list.append(f"fitted losses are {'| '.join(fitted_loss_per_ensemble)}")

        if self.initialized and self.cold_start is False:
            self.num_epochs = self.hot_start_epochs

        return loss_list

    def predict(self, data: DataLoader) -> np.ndarray:
        predicted_list = []
        for index in range(self.num_ensembles):
            with torch.no_grad():
                predicted_list.append(
                    np.concatenate(
                        [self.model[index].predict(instances).data.cpu().detach().numpy() for instances, _ in data],
                        axis=0,
                    )
                )
        return predicted_list

    # def reset(self) -> None:
    #     self.initialized = False
    #     self.num_epochs = EPOCHS
    #     self.starting_epoch = 0
    #     for index in range(self.num_samples):
    #         for name, module in self.model[index].named_children():
    #             if name not in ["sigmoid", "softmax", "relu", "dropout"]:
    #                 torch.nn.init.normal(module, mean=0, std=np.sqrt(self.variance))
    #                 module.reset_parameters()

    def reset(self) -> None:
        self.num_epochs = EPOCHS
        for index in range(self.num_ensembles):
            self._init_normal_weights(index)

    def _init_normal_weights(self, index: int) -> None:
        for name, module in self.model[index].named_children():
            if name not in ["sigmoid", "softmax", "relu", "dropout"]:
                module.reset_parameters()
        # torch.nn.init.normal(self.model[index].hidden.weight, mean=0, std=np.sqrt(self.variance))
        # torch.nn.init.normal(self.model[index].hidden.bias, mean=0, std=np.sqrt(self.variance))
        # torch.nn.init.normal(self.model[index].output.weight, mean=0, std=np.sqrt(self.variance))
        # torch.nn.init.normal(self.model[index].output.bias, mean=0, std=np.sqrt(self.variance))
