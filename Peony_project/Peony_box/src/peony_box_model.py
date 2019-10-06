import numpy as np

from typing import Callable, Any, List, Dict, Any, Optional, Union
from Peony_box.src.peony_adjusted_models.generalized_peony_box_model import (
    GeneralizedPeonyBoxModel,
)
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest


class PeonyBoxModel:
    def __init__(
        self,
        transformator: Callable[[Union[List[Dict[str, Any]], List[str]]], np.ndarray],
        acquisition_function: Optional[Callable[[np.ndarray], List[int]]],
        active_learning_step: Optional[int],
    ):
        self.random_forest_model = GeneralizedPeonyBoxModel(
            PeonyRandomForest(),
            transformator,
            acquisition_function,
            active_learning_step,
        )
