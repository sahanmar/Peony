import numpy as np
import random

from tqdm import tqdm
from typing import Any, List, Dict, Callable, Union
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def k_fold_corss_validation(
    model: Any,
    transformator: Callable[[Union[List[Dict[str, Any]], List[str]]], np.ndarray],
    validation_instances: List[Dict[str, Any]],
    validation_labels: List[Any],
    splits: int,
) -> List[Dict[str, np.ndarray]]:

    print("transforming instances for k fold cross validation...")
    validation_instances = transformator(validation_instances)
    print("transforming labels for k fold cross validation...")
    validation_labels = transformator(validation_labels)

    validation_instances, validation_labels = shuffle(
        validation_instances, validation_labels, random_state=0
    )

    true_vs_predicted: List[Dict[str, np.ndarray]] = []
    kf = KFold(n_splits=splits)
    kf.get_n_splits(validation_instances)

    print("k fold cross validation...")
    for train_index, test_index in tqdm(list(kf.split(validation_instances))):
        X_train, X_test = (
            validation_instances[train_index],
            validation_instances[test_index],
        )
        y_train, y_test = (
            validation_labels[train_index],
            validation_labels[test_index],
        )
        model.reset()
        model.training_dataset = {}
        model.fit(X_train, y_train, transformation_needed=False)
        y_predicted = model.predict(X_test, transformation_needed=False)
        true_vs_predicted.append({"true": y_test, "predicted": y_predicted})

    return true_vs_predicted
