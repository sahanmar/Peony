import numpy as np

from PeonyPackage.PeonyDb import MongoDb
from Peony_visualization.src.peony_visualization import calculate_binary_metrics
from Peony_box.src.peony_box_model import PeonyBoxModel
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from Peony_box.src.transformators.HuffPost_transformator import (
    HuffPostTransformWordEmbeddings as transformator,
)
from Peony_database.src.datasets.HuffPost_news_dataset import (
    COLLECTION_NAME as HuffPost_collection_name,
    COLLECTION_ID as HuffPost_collection_id,
)
from Peony_box.src.acquisition_functions.functions import entropy_sampling
from scipy.sparse import vstack
from sklearn.utils import shuffle

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from Peony_box.src.utils import k_fold_corss_validation, auc_metrics
from sklearn.metrics import accuracy_score


def main():
    api = MongoDb()
    sport_records = api.get_record(
        collection_name=HuffPost_collection_name,
        collection_id=HuffPost_collection_id,
        label="SPORTS",
        limit=100,
    )

    comedy_records = api.get_record(
        collection_name=HuffPost_collection_name,
        collection_id=HuffPost_collection_id,
        label="COMEDY",
        limit=100,
    )

    instances = sport_records + comedy_records
    labels = [sample["record"]["label"] for sample in sport_records + comedy_records]

    HuffPostTransform = transformator()
    HuffPostTransform.fit(instances)

    peony_model = PeonyBoxModel(
        HuffPostTransform, active_learning_step=5, acquisition_function=entropy_sampling
    )
    # peony_model.svm_model.fit(instances[50:], labels[50:])
    # indexes = peony_model.svm_model.get_learning_samples(instances[:50])

    # add_training = [instances[index] for index in indexes.tolist()]
    # add_labels = [labels[index] for index in indexes.tolist()]

    # peony_model.feed_forward_nn.add_new_learning_samples(add_training, add_labels)
    # peony_model.feed_forward_nn.fit(instances, labels)
    # predicted = peony_model.feed_forward_nn.predict(instances)

    k_fold = k_fold_corss_validation(
        peony_model.feed_forward_nn, HuffPostTransform, instances, labels, 3
    )

    print(auc_metrics(k_fold))

    scores = [
        accuracy_score(eval["true"], eval["predicted"], normalize=True)
        for eval in k_fold
    ]

    print(scores)
    print("test")


if __name__ == "__main__":
    main()
