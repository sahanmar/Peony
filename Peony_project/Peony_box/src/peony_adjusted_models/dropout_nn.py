import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from torch.optim import Optimizer
from typing import Optional, Tuple, List


NUM_SAMPLES = 10
EPOCHS_PER_SAMPLE = 50
EPOCHS = 3000
HOT_START_EPOCHS = 100
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MINI_BATCH_RATIO = 0.5
LEARNING_RATE = 0.001
WEIGHTS_VARIANCE = 0.1


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.6, inplace=False)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class PeonyDropoutFeedForwardNN:
    def __init__(
        self, hidden_size: int, num_classes: int, cold_start=False,
    ):

        self.num_samples = NUM_SAMPLES
        self.epochs_per_sample = EPOCHS_PER_SAMPLE
        self.starting_epoch = 0
        self.hot_start_epochs = HOT_START_EPOCHS

        self.model: Optional[List[NeuralNet]] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None
        self.optimizer: Optional[torch.optim.Adam] = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False
        self.mini_batch = MINI_BATCH_RATIO
        self.cold_start = cold_start
        self.variance = WEIGHTS_VARIANCE

    def fit(self, instances: np.ndarray, labels: np.ndarray) -> Optional[List[str]]:

        loss_list: List[str] = []
        self.num_of_minibatch_samples = int(instances.shape[0] * self.mini_batch)

        if self.initialized is False:
            self.model = [
                NeuralNet(instances.shape[1], self.hidden_size, self.num_classes).to(
                    DEVICE
                )
                for i in range(self.num_samples)
            ]
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(
                self.model[0].parameters(), lr=LEARNING_RATE
            )
            self.initialized = True

        try:
            instances = torch.from_numpy(instances.toarray()).float()
        except AttributeError:
            instances = torch.from_numpy(instances).float()
        labels = torch.from_numpy(labels)
        fitted_loss_per_sample: List[float] = []
        for index in range(self.num_samples):

            if self.cold_start is False:
                with torch.no_grad():
                    for param in self.model[index].parameters():
                        param.add_(torch.randn(param.size()) * self.variance)

            if index != 0:
                self.model[index].load_state_dict(self.model[0].state_dict())
                self.starting_epoch = self.num_epochs
                self.num_epochs += self.epochs_per_sample

            for epoch in range(self.starting_epoch, self.num_epochs):

                indices = np.random.choice(
                    instances.shape[0], self.num_of_minibatch_samples, replace=False
                )

                # Forward pass
                outputs = self.model[0].train()(instances[indices, :])
                loss = self.criterion(outputs, labels[indices].long())
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch == 0:
                    initial_loss_per_sample = loss.detach().numpy()
            fitted_loss_per_sample.append(loss.detach().numpy())
        loss_list.append(f"starting loss is {initial_loss_per_sample}")
        loss_list.append(
            f"fitted loss (samples mean) is {np.mean(fitted_loss_per_sample)}"
        )

        if self.initialized and self.cold_start is False:
            self.starting_epoch = 0
            self.num_epochs = self.hot_start_epochs
        else:
            self.starting_epoch = 0
            self.num_epochs = EPOCHS

        return loss_list

    def predict(self, instances: np.ndarray) -> np.ndarray:
        try:
            instances = torch.from_numpy(instances.toarray()).float()
        except AttributeError:
            instances = torch.from_numpy(instances).float()
        predicted_list = []
        for index in range(self.num_samples):
            with torch.no_grad():
                outputs = self.model[index](instances)
                _, predicted = torch.max(outputs.data, 1)
                predicted_list.append(predicted.detach().numpy())
        return predicted_list

    def reset(self) -> None:
        self.initialized = False
        self.num_epochs = EPOCHS
        self.starting_epoch = 0
        for index in range(self.num_samples):
            self.model[index].hidden.reset_parameters()
            self.model[index].output.reset_parameters()
