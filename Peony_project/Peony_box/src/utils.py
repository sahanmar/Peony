import numpy as np
import random

from tqdm import tqdm
from typing import Any, List, Dict, Callable, Union, Tuple
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import auc, roc_curve
from Peony_box.src.transformators.generalized_transformator import Transformator


def transform_label_to_binary(
    true_vs_predicted: List[Dict[str, np.ndarray]]
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:

    unique_values = np.unique(true_vs_predicted[0]["true"])
    if len(unique_values) > 2:
        raise Exception("This is not binary classification")
    if len(unique_values) != 2:
        mapped_to_0 = unique_values[0]
        print(f"Label {mapped_to_0} in mapped to 0, another label is mapped to 1")
    else:
        mapped_to_0 = unique_values[0]
        mapped_to_1 = unique_values[1]
        print(f"Label {mapped_to_0} in mapped to 0, label {mapped_to_1} in mapped to 1")
    for record in true_vs_predicted:
        for index in range(len(record["true"])):
            record["true"][index] = 0 if record["true"][index] == mapped_to_0 else 1
            record["predicted"][index] = (
                0 if record["predicted"][index] == mapped_to_0 else 1
            )

    return (true_vs_predicted, unique_values)


def auc_metrics(
    true_vs_predicted: List[Dict[str, np.ndarray]], label_to_binary: bool = True
) -> list:

    if label_to_binary:
        true_vs_predicted, unique_values = transform_label_to_binary(true_vs_predicted)

    aucs = []
    for index, record in enumerate(true_vs_predicted):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(record["true"], record["predicted"])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    return aucs


def k_fold_corss_validation(
    model: Any,
    transformator: Transformator,
    validation_instances: List[Dict[str, Any]],
    validation_labels: List[Any],
    splits: int,
    transform_label_to_binary: bool = False,
) -> List[Dict[str, np.ndarray]]:

    model_output: list = []

    print("transforming instances for k fold cross validation...")
    validation_instances = transformator.transform_instances(validation_instances)
    print("transforming labels for k fold cross validation...")
    validation_labels = transformator.transform_labels(validation_labels)

    validation_instances, validation_labels = shuffle(
        validation_instances, validation_labels, random_state=0
    )

    if transform_label_to_binary:
        unique_values = np.unique(validation_labels)
        if unique_values is not None:
            validation_labels = np.asarray(
                [0 if x == unique_values[0] else 1 for x in validation_labels]
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
