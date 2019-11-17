from PeonyPackage.PeonyDb import MongoDb
from Peony_box.active_learning_simulation.utils import active_learning_simulation
from Peony_box.src.transformators.HuffPost_transformator import (
    HuffPostTransform as transformator,
)
from Peony_box.src.transformators.HuffPost_transformator import (
    HuffPostTransformWordEmbeddings as word_embed_transformator,
)
from Peony_database.src.datasets.HuffPost_news_dataset import (
    COLLECTION_NAME as HuffPost_collection_name,
    COLLECTION_ID as HuffPost_collection_id,
)
from Peony_box.src.acquisition_functions.functions import entropy_sampling
from Peony_visualization.src.peony_visualization import visualize_two_auc_evolutions

from sklearn.utils import shuffle
import numpy as np


def main():

    api = MongoDb()

    sport_records = api.get_record(
        collection_name=HuffPost_collection_name,
        collection_id=HuffPost_collection_id,
        label="SPORTS",
        limit=500,
    )

    comedy_records = api.get_record(
        collection_name=HuffPost_collection_name,
        collection_id=HuffPost_collection_id,
        label="COMEDY",
        limit=500,
    )

    # Define model specifications
    model_1 = "feed_forward_nn_fast_text_embeddings"
    model_2 = "feed_forward_nn_fast_text_embeddings"
    algorithm = "nn"
    acquisition_function_1 = "random"
    acquisition_function_2 = "entropy"
    active_learning_loops = 10
    active_learning_step = 1
    max_active_learning_iters = 2
    initial_training_data_size = 10
    validation_data_size = 1000
    category_1 = "SPORTS"
    category_2 = "COMEDY"
    transformation_needed = True

    instances = sport_records + comedy_records
    labels = [sample["record"]["label"] for sample in sport_records + comedy_records]

    instances_from_db, labels_from_db = shuffle(instances, labels, random_state=0)

    # HuffPostTransform = word_embed_transformator()
    HuffPostTransform = transformator()
    HuffPostTransform.fit(instances_from_db)

    if transformation_needed:
        instances = instances_from_db
        labels = labels_from_db
    else:
        try:
            instances = np.asarray(
                HuffPostTransform.transform_instances(instances_from_db).todense()
            )
        except:
            instances = np.asarray(
                HuffPostTransform.transform_instances(instances_from_db)
            )
        labels = np.asarray(HuffPostTransform.transform_labels(labels_from_db))

    # Get AUC results from an active learning simulation
    auc_active_learning_random_10_runs_nn = active_learning_simulation(
        HuffPostTransform,
        None,
        active_learning_loops,
        max_active_learning_iters,
        active_learning_step,
        algorithm,
        instances,
        labels,
        initial_training_data_size,
        transformation_needed,
    )

    # Pack specifications and resutls to the list for uploading to Peony Database
    list_to_upload = [
        model_1,
        acquisition_function_1,
        active_learning_loops,
        active_learning_step,
        max_active_learning_iters,
        initial_training_data_size,
        validation_data_size,
        category_1,
        category_2,
        auc_active_learning_random_10_runs_nn,
    ]

    # Upload results to Peony Database
    # api.load_model_results(*list_to_upload)

    # Get AUC results from an active learning simulation
    auc_active_learning_entropy_10_runs_nn = active_learning_simulation(
        HuffPostTransform,
        entropy_sampling,
        active_learning_loops,
        max_active_learning_iters,
        active_learning_step,
        algorithm,
        instances,
        labels,
        initial_training_data_size,
        transformation_needed,
    )

    # Pack specifications and resutls to the list for uploading to Peony Database
    list_to_upload = [
        model_2,
        acquisition_function_2,
        active_learning_loops,
        active_learning_step,
        max_active_learning_iters,
        initial_training_data_size,
        validation_data_size,
        category_1,
        category_2,
        auc_active_learning_entropy_10_runs_nn,
    ]

    # Upload results to Peony Database
    # api.load_model_results(*list_to_upload)

    visualize_two_auc_evolutions(
        auc_active_learning_random_10_runs_nn, auc_active_learning_entropy_10_runs_nn
    )


if __name__ == "__main__":
    main()
