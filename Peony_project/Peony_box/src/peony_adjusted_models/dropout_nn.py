import numpy as np
import torch
import torch.nn as nn

from Peony_box.src.peony_adjusted_models.neural_nets_architecture import (
    NeuralNet,
    NeuralNetLSTM,
)

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

NUM_SAMPLES = 10
EPOCHS_PER_SAMPLE = 50
EPOCHS = 3000
HOT_START_EPOCHS = 100
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
WEIGHTS_VARIANCE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neural_network = NeuralNet

# NUM_SAMPLES = 10
# EPOCHS_PER_SAMPLE = 1
# EPOCHS = 180
# HOT_START_EPOCHS = 50
# # Device configuration
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LEARNING_RATE = 0.001
# WEIGHTS_VARIANCE = 0.10
# neural_network = NeuralNetLSTM


class PeonyDropoutFeedForwardNN:
    def __init__(self, hidden_size: int, num_classes: int, cold_start=False, dropout_in_eval=True):

        self.num_samples = NUM_SAMPLES
        self.epochs_per_sample = EPOCHS_PER_SAMPLE
        self.starting_epoch = 0
        self.hot_start_epochs = HOT_START_EPOCHS

        self.model: Optional[List[nn.Module]] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None
        self.optimizer: Optional[torch.optim.Adam] = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False
        self.cold_start = cold_start
        self.variance = WEIGHTS_VARIANCE
        self.dropout_in_eval = dropout_in_eval

    def fit(self, data: DataLoader, features_size: int) -> Optional[List[str]]:

        loss_list: List[str] = []

        if self.initialized is False:
            self.model = [
                neural_network(features_size, self.hidden_size, self.num_classes, dropout=0.2).to(DEVICE)
                for i in range(self.num_samples)
            ]
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model[0].parameters(), lr=LEARNING_RATE)
            self.initialized = True

        fitted_loss_per_sample: List[float] = []
        for index in range(self.num_samples):

            if self.cold_start is False:
                with torch.no_grad():
                    for param in self.model[index].parameters():
                        param.add_(torch.randn(param.size()).to(DEVICE) * self.variance)

            if index != 0:
                self.model[index].load_state_dict(self.model[0].state_dict())
                self.starting_epoch = self.num_epochs
                self.num_epochs += self.epochs_per_sample

            for epoch in range(self.starting_epoch, self.num_epochs):

                for instances, labels in data:

                    # Forward pass
                    self.optimizer.zero_grad()

                    outputs = self.model[0].train()(instances)

                    loss = self.criterion(outputs, labels.to(DEVICE))
                    # Backward and optimize
                    loss.backward()
                    self.optimizer.step()

                    if epoch == 0:
                        initial_loss_per_sample = loss.cpu().detach().numpy()

            fitted_loss_per_sample.append(loss.cpu().detach().numpy())
        loss_list.append(f"starting loss is {initial_loss_per_sample}")
        loss_list.append(f"fitted loss (samples mean) is {np.mean(fitted_loss_per_sample)}")

        if self.initialized and self.cold_start is False:
            self.starting_epoch = 0
            self.num_epochs = self.hot_start_epochs
        else:
            self.starting_epoch = 0
            self.num_epochs = EPOCHS

        return loss_list

    def predict(self, data: DataLoader) -> np.ndarray:
        predicted_list = []
        for index in range(self.num_samples):
            if not self.dropout_in_eval:
                self.model[index].eval()
            with torch.no_grad():
                predicted_list.append(
                    np.concatenate(
                        [self.model[index].predict(instances).data.cpu().detach().numpy() for instances, _ in data],
                        axis=0,
                    )
                )

        return predicted_list

    def reset(self) -> None:
        self.initialized = False
        self.num_epochs = EPOCHS
        self.starting_epoch = 0
        for index in range(self.num_samples):
            for name, module in self.model[index].named_children():
                if name not in ["sigmoid", "softmax", "relu", "dropout"]:
                    module.reset_parameters()
