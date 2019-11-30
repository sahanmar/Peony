from Peony_box.src.peony_box_model import PeonyBoxModel

from Peony_box.src.utils import k_fold_corss_validation, auc_metrics

import numpy as np
import multiprocessing as mp
import argparse

from IPython.utils import io
from tqdm import tqdm


def reset_validation_data(testing_instances, testing_labels, new_training_indices):
    new_training_indices = new_training_indices.tolist()
    training_instances = [testing_instances[index] for index in new_training_indices]
    training_labels = [testing_labels[index] for index in new_training_indices]
    testing_instances = np.delete(testing_instances, new_training_indices, axis=0)
    testing_labels = np.delete(testing_labels, new_training_indices, axis=None)

    return training_instances, training_labels, testing_instances, testing_labels


def active_learning_simulation(
    transformator,
    acquisition_function,
    active_learning_loops,
    max_active_learning_iters,
    active_learning_step,
    model,
    instances,
    labels,
    initial_training_data_size,
    transformation_needed,
):

    # pool = mp.Pool(mp.cpu_count())

    # Repeat experiment for statistical validation
    # Return auc results for all runs and different active learning iteration

    # result = [pool.apply(active_learning_simulation_round, args = (transformator, acquisition_function, max_active_learning_iters, active_learning_step, model)) for _ in range(active_learning_loops)]

    # pool.close()

    return [
        active_learning_simulation_round(
            transformator,
            acquisition_function,
            max_active_learning_iters,
            active_learning_step,
            model,
            instances,
            labels,
            initial_training_data_size,
            transformation_needed,
        )
        for _ in tqdm(range(active_learning_loops))
    ]


def active_learning_simulation_round(
    transformator,
    acquisition_function,
    max_active_learning_iters,
    active_learning_step,
    model,
    instances,
    labels,
    initial_training_data_size,
    transformation_needed,
):
    auc_active_learning_runs = []
    # Data preparation
    auc_active_learning = []

    training_instances = instances[:initial_training_data_size]
    training_labels = labels[:initial_training_data_size]

    testing_instances = instances[initial_training_data_size:]
    testing_labels = labels[initial_training_data_size:]

    # Active Learning Pipeline Run
    peony_model = PeonyBoxModel(
        transformator,
        active_learning_step=active_learning_step,
        acquisition_function=acquisition_function,
    )
    with io.capture_output() as captured:  # suppressing output
        # Fit model with very little set of training data
        if model == "svm":
            peony_model.svm_model.fit(
                training_instances, training_labels, transformation_needed
            )
        elif model == "nn":
            peony_model.feed_forward_nn.fit(
                training_instances, training_labels, transformation_needed
            )
        elif model == "bayesian_sgld":
            peony_model.bayesian_sgld_nn.fit(
                training_instances, training_labels, transformation_needed
            )
        else:
            peony_model.random_forest_model.fit(
                training_instances, training_labels, transformation_needed
            )

    # Start active learning loop
    for index in range(max_active_learning_iters):
        with io.capture_output() as captured:  # suppressing output

            # predict the dataset complement for choosing next training data
            if model == "svm":
                predicted = peony_model.svm_model.predict(
                    testing_instances, transformation_needed
                )
            elif model == "nn":
                predicted = peony_model.feed_forward_nn.predict(
                    testing_instances, transformation_needed
                )
            elif model == "bayesian_sgld":
                predicted = peony_model.bayesian_sgld_nn.predict(
                    testing_instances, transformation_needed
                )
            else:
                predicted = peony_model.random_forest_model.predict(
                    testing_instances, transformation_needed
                )

            if transformation_needed:
                labels_for_auc = transformator.transform_labels(testing_labels[:])
            else:
                labels_for_auc = list(testing_labels[:])
            auc_active_learning.append(
                auc_metrics([{"true": labels_for_auc, "predicted": predicted}])
            )

            # Get indices based on acquisition function
            if model == "svm":
                indices = peony_model.svm_model.get_learning_samples(
                    testing_instances, transformation_needed
                )
            elif model == "nn":
                indices = peony_model.feed_forward_nn.get_learning_samples(
                    testing_instances, transformation_needed
                )
            elif model == "bayesian_sgld":
                indices = peony_model.bayesian_sgld_nn.get_learning_samples(
                    testing_instances, transformation_needed
                )
            else:
                indices = peony_model.random_forest_model.get_learning_samples(
                    testing_instances, transformation_needed
                )

            # Reset validation dataset (add training data, remove testing data)
            (
                training_instances,
                training_labels,
                testing_instances,
                testing_labels,
            ) = reset_validation_data(testing_instances, testing_labels, indices)

            # Add new learning samples to the model and retrain
            if model == "svm":
                peony_model.svm_model.add_new_learning_samples(
                    training_instances, training_labels, transformation_needed
                )
            elif model == "nn":
                peony_model.feed_forward_nn.add_new_learning_samples(
                    training_instances, training_labels, transformation_needed
                )
            elif model == "bayesian_sgld":
                peony_model.bayesian_sgld_nn.add_new_learning_samples(
                    training_instances, training_labels, transformation_needed
                )
            else:
                peony_model.random_forest_model.add_new_learning_samples(
                    training_instances, training_labels, transformation_needed
                )

    return auc_active_learning
