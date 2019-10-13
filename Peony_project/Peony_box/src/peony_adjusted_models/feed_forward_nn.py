import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Tuple, List

NUM_ENSEMBLES = 10
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
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


class PeonyFeedForwardNN:
    def __init__(self, hidden_size, num_classes):

        self.num_ensembles = NUM_ENSEMBLES

        self.model = None
        self.criterion = None
        self.optimizer = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = 300
        self.initialized = False

    def fit(self, instances: np.ndarray, labels: np.ndarray) -> Optional[List[str]]:

        loss_list: List[str] = []

        if self.initialized is False:
            self.model = [
                NeuralNet(instances.shape[1], self.hidden_size, self.num_classes).to(
                    DEVICE
                )
                for i in range(self.num_ensembles)
            ]
            self.criterion = [nn.CrossEntropyLoss() for i in range(self.num_ensembles)]
            self.optimizer = [
                torch.optim.Adam(self.model[i].parameters(), lr=LEARNING_RATE)
                for i in range(self.num_ensembles)
            ]
            self.initialized = True

        instances = torch.from_numpy(instances.toarray()).float()
        labels = torch.from_numpy(labels)

        for index in range(self.num_ensembles):
            initial_loss_per_ensemble: List[float] = []
            fitted_loss_per_ensemble: List[float] = []
            for epoch in range(self.num_epochs):
                # Forward pass
                outputs = self.model[index](instances)
                loss = self.criterion[index](outputs, labels)
                # Backward and optimize
                self.optimizer[index].zero_grad()
                loss.backward()
                self.optimizer[index].step()

                if epoch == 0:
                    initial_loss_per_ensemble.append(loss.detach().numpy())
            fitted_loss_per_ensemble.append(loss.detach().numpy())
        loss_list.append(
            f"starting loss (ensembles mean) is {np.mean(initial_loss_per_ensemble)}"
        )
        loss_list.append(
            f"fitted loss (ensembles mean) is {np.mean(fitted_loss_per_ensemble)}"
        )

        return loss_list

    def predict(self, instances: np.ndarray) -> np.ndarray:
        instances = torch.from_numpy(instances.toarray()).float()
        predicted_list = []
        for index in range(self.num_ensembles):
            with torch.no_grad():
                outputs = self.model[index](instances)
                _, predicted = torch.max(outputs.data, 1)
                predicted_list.append(predicted.detach().numpy())
        return predicted_list

    def reset(self) -> None:
        for index in range(self.num_ensembles):
            self.model[index].hidden.reset_parameters()
            self.model[index].output.reset_parameters()
