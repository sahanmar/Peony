import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader


from Peony_box.src.peony_adjusted_models.neural_nets_architecture import (
    NeuralNet,
    NeuralNetLSTM,
)


NUM_ENSEMBLES = 10
EPOCHS = 2000
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001

neural_network = NeuralNet


class PeonyFeedForwardNN:
    def __init__(
        self, hidden_size: int, num_classes: int, rand_sample_ratio: int, num_ensembles: int = NUM_ENSEMBLES
    ):

        self.num_ensembles = num_ensembles

        self.model: Optional[List[NeuralNet]] = None
        self.criterion: Optional[List[nn.CrossEntropyLoss]] = None
        self.optimizer: Optional[List[torch.optim.SGD]] = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False
        self.rand_sample_ratio = rand_sample_ratio

    def fit(self, data: DataLoader, features_size: int) -> Optional[List[str]]:

        loss_list: List[str] = []

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
            self.initialized = True

        initial_loss_per_ensemble: List[float] = []
        fitted_loss_per_ensemble: List[float] = []
        for index in range(self.num_ensembles):
            for epoch in range(self.num_epochs):

                for instances, labels in data:

                    # Forward pass
                    self.optimizer[index].zero_grad()
                    outputs = self.model[index].train()(instances)
                    loss = self.criterion[index](outputs, labels)
                    # Backward and optimize
                    loss.backward()
                    self.optimizer[index].step()

                    if epoch == 0:
                        initial_loss_per_ensemble.append(loss.detach().numpy())
            fitted_loss_per_ensemble.append(loss.detach().numpy())
        loss_list.append(f"starting loss (ensembles mean) is {np.mean(initial_loss_per_ensemble)}")
        loss_list.append(f"fitted loss (ensembles mean) is {np.mean(fitted_loss_per_ensemble)}")

        if self.initialized:
            self.num_epochs = 20

        return loss_list

    def predict(self, data: DataLoader) -> np.ndarray:
        predicted_list = []
        for index in range(self.num_ensembles):
            with torch.no_grad():
                predicted_list.append(
                    np.concatenate(
                        [self.model[index].predict(instances).data.detach().numpy() for instances, _ in data],
                        axis=0,
                    )
                )
        return predicted_list

    def reset(self) -> None:
        self.initialized = False
        self.num_epochs = EPOCHS
        for index in range(self.num_ensembles):
            self.model[index].hidden.reset_parameters()
            self.model[index].output.reset_parameters()
