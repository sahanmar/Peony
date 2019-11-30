import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from torch.optim import Optimizer
from typing import Optional, Tuple, List


NUM_SAMPLES = 50
EPOCHS_PER_SAMPLE = 20
EPOCHS = 2000
STEPS_TO_BURN = 2000
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MINI_BATCH_RATIO = 0.5
LEARNING_RATE = 0.01


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


class PeonySGLDFeedForwardNN:
    def __init__(self, hidden_size: int, num_classes: int):

        self.num_samples = NUM_SAMPLES
        self.epochs_per_sample = EPOCHS_PER_SAMPLE
        self.starting_epoch = 0

        self.model: Optional[List[NeuralNet]] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None
        self.optimizer: Optional[List[pysgmcmc.optimizers.sgld.SGLD]] = None

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False
        self.mini_batch = MINI_BATCH_RATIO
        self.steps_to_burn = STEPS_TO_BURN

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
            self.optimizer = SGLD(
                self.model[0].parameters(),
                lr=LEARNING_RATE,
                precondition_decay_rate=0.95,
                num_burn_in_steps=self.steps_to_burn,
            )
            self.initialized = True

        try:
            instances = torch.from_numpy(instances.toarray()).float()
        except AttributeError:
            instances = torch.from_numpy(instances).float()
        labels = torch.from_numpy(labels)
        for index in range(self.num_samples):
            fitted_loss_per_sample: List[float] = []
            if index != 0:
                self.model[index].load_state_dict(self.model[0].state_dict())
                self.starting_epoch = self.num_epochs
                self.num_epochs += self.epochs_per_sample

            for epoch in range(self.starting_epoch, self.num_epochs):

                indices = np.random.choice(
                    instances.shape[0], self.num_of_minibatch_samples, replace=False
                )

                # Forward pass
                outputs = self.model[index](instances[indices, :])
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


# Pytorch Port of a previous tensorflow implementation in `tensorflow_probability`:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics.md

# Corrected and changed by github.com/sahanmar


class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        precondition_decay_rate=0.95,
        num_pseudo_batches=1,
        num_burn_in_steps=3000,
        diagonal_bias=1e-8,
    ) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr,
            precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    lr = lr / (state["iteration"] - group["num_burn_in_steps"]) + 0.05
                    sigma = torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = 1.0 / torch.sqrt(momentum + group["diagonal_bias"])

                scaled_grad = preconditioner * gradient + torch.normal(
                    mean=torch.zeros_like(gradient),
                    std=torch.ones_like(gradient) * sigma,
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss
