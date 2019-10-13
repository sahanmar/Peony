import numpy as np

from typing import Callable, Any, List, Dict, Any, Optional, Union
from Peony_box.src.peony_adjusted_models.generalized_peony_box_model import (
    GeneralizedPeonyBoxModel,
)
from Peony_box.src.transformators.generalized_transformator import Transformator
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from Peony_box.src.peony_adjusted_models.svm_model import PeonySVM
from Peony_box.src.peony_adjusted_models.feed_forward_nn import PeonyFeedForwardNN
from Peony_box.src.acquisition_functions.functions import random_sampling


class PeonyBoxModel:
    def __init__(
        self,
        transformator: Transformator,
        acquisition_function: Callable[[np.ndarray], np.ndarray] = random_sampling,
        active_learning_step: int = 1,
    ):
        self.random_forest_model = GeneralizedPeonyBoxModel(
            model=PeonyRandomForest(),
            transformator=transformator,
            acquisition_function=acquisition_function,
            active_learning_step=active_learning_step,
        )
        self.svm_model = GeneralizedPeonyBoxModel(
            model=PeonySVM(),
            transformator=transformator,
            acquisition_function=acquisition_function,
            active_learning_step=active_learning_step,
        )
        self.feed_forward_nn = GeneralizedPeonyBoxModel(
            model=PeonyFeedForwardNN(hidden_size=100, num_classes=41),
            transformator=transformator,
            acquisition_function=acquisition_function,
            active_learning_step=active_learning_step,
        )
