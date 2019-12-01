import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from typing import Optional, Tuple, List

NUM_ENSEMBLES = 10
EPOCHS = 2000
HOT_START_EPOCHS = 500
WEIGHTS_VARIANCE = 0.4
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
        self.num_of_samples = 0

        self.model: Optional[List[NeuralNet]] = None
        self.criterion: Optional[List[nn.CrossEntropyLoss]] = None
        self.optimizer: Optional[List[torch.optim.SGD]] = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False
        self.rand_sample_ratio = rand_sample_ratio
        self.variance = WEIGHTS_VARIANCE
        self.loss_sequence: List[List[float]] = []
        self.cold_start = cold_start
        self.hot_start_epochs = HOT_START_EPOCHS

    def fit(self, instances: np.ndarray, labels: np.ndarray) -> Optional[List[str]]:

        loss_list: List[str] = []
        self.loss_sequence: List[List[float]] = []
        self.num_of_samples = int(instances.shape[0] * self.rand_sample_ratio)

        if self.initialized is False:
            self.model = [
                NeuralNet(instances.shape[1], self.hidden_size, self.num_classes).to(
                    DEVICE
                )
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

        try:
            instances = torch.from_numpy(instances.toarray()).float()
        except AttributeError:
            instances = torch.from_numpy(instances).float()
        labels = torch.from_numpy(labels)
        initial_loss_per_ensemble: List[str] = []
        fitted_loss_per_ensemble: List[str] = []
        for index in range(self.num_ensembles):

            if self.cold_start is False:
                with torch.no_grad():
                    for param in self.model[index].parameters():
                        param.add_(torch.randn(param.size()) * self.variance)

            loss_sequence_per_ensemble: List[float] = []

            indices = np.random.choice(
                instances.shape[0], self.num_of_samples, replace=False
            )
            for epoch in range(self.num_epochs):
                # Forward pass
                outputs = self.model[index](instances[indices, :])
                loss = self.criterion[index](outputs, labels[indices].long())
                loss_sequence_per_ensemble.append(float(loss.detach().numpy()))
                # Backward and optimize
                self.optimizer[index].zero_grad()
                loss.backward()
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

    def predict(self, instances: np.ndarray) -> np.ndarray:
        try:
            instances = torch.from_numpy(instances.toarray()).float()
        except AttributeError:
            instances = torch.from_numpy(instances).float()
        predicted_list = []
        for index in range(self.num_ensembles):
            with torch.no_grad():
                outputs = self.model[index](instances)
                _, predicted = torch.max(outputs.data, 1)
                predicted_list.append(predicted.detach().numpy())
        return predicted_list

    def reset(self) -> None:
        self.initialized = False
        self.num_epochs = EPOCHS
        for index in range(self.num_ensembles):
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
