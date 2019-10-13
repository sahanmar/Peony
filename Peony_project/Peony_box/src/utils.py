import numpy as np
import random

from tqdm import tqdm
from typing import Any, List, Dict, Callable, Union
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from Peony_box.src.transformators.generalized_transformator import Transformator


def k_fold_corss_validation(
    model: Any,
    transformator: Transformator,
    validation_instances: List[Dict[str, Any]],
    validation_labels: List[Any],
    splits: int,
) -> List[Dict[str, np.ndarray]]:

    model_output: list = []

    transformator = transformator()
    print("transforming instances for k fold cross validation...")
    validation_instances = transformator.transform_instances(validation_instances)
    print("transforming labels for k fold cross validation...")
    validation_labels = transformator.transform_labels(validation_labels)

    validation_instances, validation_labels = shuffle(
        validation_instances, validation_labels, random_state=0
    )

    true_vs_predicted: List[Dict[str, np.ndarray]] = []
    kf = KFold(n_splits=splits)
    kf.get_n_splits(validation_instances)

    print("k fold cross validation...")
    splitted = list(kf.split(validation_instances))
    for train_index, test_index in tqdm(splitted):
        X_train, X_test = (
            validation_instances[train_index],
            validation_instances[test_index],
        )
        y_train, y_test = (
            validation_labels[train_index],
            validation_labels[test_index],
        )
        model.training_dataset = {}

        output = model.fit(X_train, y_train, transformation_needed=False)
        if output:
            model_output.extend(output)

        y_predicted = model.predict(X_test, transformation_needed=False)
        true_vs_predicted.append({"true": y_test, "predicted": np.round(y_predicted)})
        model.reset()

    if model_output != []:
        print("\n".join(" , ".join(output) for output in model_output))

    return true_vs_predicted
