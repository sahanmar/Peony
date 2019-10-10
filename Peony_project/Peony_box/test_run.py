import numpy as np

from PeonyPackage.PeonyDb import MongoDb
from Peony_visualization.src.peony_visualization import calculate_binary_metrics
from Peony_box.src.peony_box_model import PeonyBoxModel
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest
from Peony_box.src.transformators import HuffPost_transformator
from Peony_database.src.datasets.HuffPost_news_dataset import (
    COLLECTION_NAME as HuffPost_collection_name,
    COLLECTION_ID as HuffPost_collection_id,
)
from scipy.sparse import vstack
from sklearn.utils import shuffle

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from Peony_box.src.utils import k_fold_corss_validation
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

    peony_model = PeonyBoxModel(HuffPost_transformator.transform, None, None)
    peony_model.random_forest_model.fit(instances, labels)

    k_fold = k_fold_corss_validation(
        peony_model.random_forest_model,
        HuffPost_transformator.transform,
        instances,
        labels,
        8,
    )

    scores = [
        accuracy_score(eval["true"], eval["predicted"], normalize=True)
        for eval in k_fold
    ]

    calculate_binary_metrics(k_fold)

    print(scores)


if __name__ == "__main__":
    main()
