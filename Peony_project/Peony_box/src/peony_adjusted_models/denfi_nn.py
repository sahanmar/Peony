import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

NUM_ENSEMBLES = 10
EPOCHS = 2000
HOT_START_EPOCHS = 500
WEIGHTS_VARIANCE = 0.2
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


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
                NeuralNet(features_size, self.hidden_size, self.num_classes).to(DEVICE)
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
                        param.add_(torch.randn(param.size()) * self.variance)

            for epoch in range(self.num_epochs):

                for instances, labels in data:
                    # Forward pass
                    outputs = self.model[index](instances)
                    loss = self.criterion[index](outputs, labels)
                    loss_sequence_per_ensemble.append(float(loss.detach().numpy()))
                    # Backward and optimize
                    self.optimizer[index].zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer[index].step()

                    if epoch == 0:
                        initial_loss_per_ensemble.append(str(loss.detach().numpy()))

            self.loss_sequence.append(loss_sequence_per_ensemble)
            fitted_loss_per_ensemble.append(str(loss.detach().numpy()))

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
                    [
                        res
                        for instances, _ in data
                        for res in torch.max(self.model[index](instances).data, 1)[1]
                        .detach()
                        .numpy()
                    ]
                )
        return predicted_list

    def reset(self) -> None:
        self.num_epochs = EPOCHS
        for index in range(self.num_ensembles):
            self._init_normal_weights(index)

    def _init_normal_weights(self, index: int) -> None:
        torch.nn.init.normal(
            self.model[index].hidden.weight, mean=0, std=np.sqrt(self.variance)
        )
        torch.nn.init.normal(
            self.model[index].hidden.bias, mean=0, std=np.sqrt(self.variance)
        )
        torch.nn.init.normal(
            self.model[index].output.weight, mean=0, std=np.sqrt(self.variance)
        )
        torch.nn.init.normal(
            self.model[index].output.bias, mean=0, std=np.sqrt(self.variance)
        )
