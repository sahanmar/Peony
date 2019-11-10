import numpy as np

from typing import Callable, Any, List, Dict, Any, Optional, Union
from Peony_box.src.peony_adjusted_models.generalized_peony_box_model import (
    GeneralizedPeonyBoxModel,
)
from Peony_box.src.transformators.generalized_transformator import Transformator
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from Peony_box.src.peony_adjusted_models.svm_model import PeonySVM
from Peony_box.src.peony_adjusted_models.feed_forward_nn import PeonyFeedForwardNN
from Peony_box.src.peony_adjusted_models.pymc3_nn import PeonyPymc3NN

RAND_SAMPLES_RATIO = 0.7


class PeonyBoxModel:
    def __init__(
        self,
        transformator: Transformator,
        acquisition_function: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        greedy_coef_decay: Optional[Callable[[int], float]] = None,
        active_learning_step: int = 1,
    ):
        self.random_forest_model = GeneralizedPeonyBoxModel(
            model=PeonyRandomForest(rand_sample_ratio=RAND_SAMPLES_RATIO),
            transformator=transformator,
            active_learning_step=active_learning_step,
            acquisition_function=acquisition_function,
            greedy_coef_decay=greedy_coef_decay,
        )
        self.svm_model = GeneralizedPeonyBoxModel(
            model=PeonySVM(rand_sample_ratio=RAND_SAMPLES_RATIO),
            transformator=transformator,
            active_learning_step=active_learning_step,
            acquisition_function=acquisition_function,
            greedy_coef_decay=greedy_coef_decay,
        )
        self.feed_forward_nn = GeneralizedPeonyBoxModel(
            model=PeonyFeedForwardNN(
                hidden_size=100, num_classes=41, rand_sample_ratio=RAND_SAMPLES_RATIO
            ),
            transformator=transformator,
            active_learning_step=active_learning_step,
            acquisition_function=acquisition_function,
            greedy_coef_decay=greedy_coef_decay,
        )
        self.bayesian_nn = GeneralizedPeonyBoxModel(
            model=PeonyPymc3NN(
                hidden_size=100, num_classes=2, rand_sample_ratio=RAND_SAMPLES_RATIO
            ),
            transformator=transformator,
            active_learning_step=active_learning_step,
            acquisition_function=acquisition_function,
            greedy_coef_decay=greedy_coef_decay,
        )
