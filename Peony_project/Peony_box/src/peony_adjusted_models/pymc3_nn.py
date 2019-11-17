import theano
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np

from warnings import filterwarnings
from typing import Optional, Tuple, List, Callable

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
filterwarnings("ignore")

EPOCHS = 60000
floatX = theano.config.floatX


class PeonyPymc3NN:
    def __init__(self, hidden_size: int, num_classes: int):
        self.n_hidden = hidden_size
        self.num_classes = num_classes
        self.num_epochs = EPOCHS
        self.initialized = False

    def fit(self, instances: np.ndarray, labels: np.ndarray) -> Optional[List[str]]:

        self.model = self._construct_nn(instances, labels)
        with self.model:
            inference = pm.ADVI()
            self.approx = pm.fit(n=EPOCHS, method=inference)

        self.sample_proba = self._sample_probability(instances)

        return None

    def predict(self, instances: np.ndarray) -> List[np.ndarray]:

        return self.sample_proba(pm.floatX(np.transpose(instances, axes=None)).T, 500)

    def _sample_probability(
        self, ann_input: np.ndarray
    ) -> Callable[[np.ndarray, int], theano.function]:
        # create symbolic input
        x = T.matrix("X")
        # symbolic number of samples is supported, we build vectorized posterior on the fly
        n = T.iscalar("n")
        # Do not forget test_values or set theano.config.compute_test_value = 'off'
        x.tag.test_value = np.empty_like(ann_input[:10])
        n.tag.test_value = 100
        _sample_proba = self.approx.sample_node(
            self.model.out.distribution.p,
            size=n,
            more_replacements={self.model["ann_input"]: x},
        )
        # It is time to compile the function
        # No updates are needed for Approximation random generator
        # Efficient vectorized form of sampling is used
        return theano.function([x, n], _sample_proba)

    def _construct_nn(self, ann_input: np.ndarray, ann_output: np.ndarray) -> pm.Model:

        input_size = ann_input.shape[1]
        total_size = ann_output.shape[0]

        # Initialize random weights between each layer
        init_1 = np.random.randn(input_size, self.n_hidden).astype(floatX)
        init_2 = np.random.randn(self.n_hidden, self.n_hidden).astype(floatX)
        init_out = np.random.randn(self.n_hidden).astype(floatX)

        with pm.Model() as neural_network:

            ann_input = pm.Data("ann_input", ann_input.astype(floatX))
            ann_output = pm.Data("ann_output", ann_output.astype(floatX))

            # Weights from input to hidden layer
            weights_in_1 = pm.Normal(
                "w_in_1", 0, sigma=1, shape=(input_size, self.n_hidden), testval=init_1
            )

            # Weights from 1st to 2nd layer
            weights_1_2 = pm.Normal(
                "w_1_2",
                0,
                sigma=1,
                shape=(self.n_hidden, self.n_hidden),
                testval=init_2,
            )

            # Weights from hidden layer to output
            weights_2_out = pm.Normal(
                "w_2_out", 0, sigma=1, shape=(self.n_hidden,), testval=init_out
            )

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=total_size,  # IMPORTANT for minibatches
            )
        return neural_network

    def reset(self) -> None:
        pass
