import numpy as np
import time

from PeonyPackage.PeonyDb import MongoDb
from Peony_visualization.src.peony_visualization import calculate_binary_metrics
from Peony_box.src.peony_box_model import PeonyBoxModel
from Peony_box.src.peony_adjusted_models.random_trees_model import PeonyRandomForest

from Peony_box.src.transformators.HuffPost_transformator import (
    # RoBERTaWordEmbeddings as transformator,
    FastTextWordEmbeddings as transformator,
)

# from Peony_database.src.datasets.Tweets_emotions_dataset import (
#     COLLECTION_NAME,
#     COLLECTION_ID,
# )

from Peony_database.src.datasets.HuffPost_news_dataset import (
    COLLECTION_NAME,
    COLLECTION_ID,
)

from Peony_box.src.acquisition_functions.functions import entropy_sampling, batch_bald, hac_sampling
from scipy.sparse import vstack
from sklearn.utils import shuffle

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from Peony_box.src.utils import k_fold_corss_validation, auc_metrics
from sklearn.metrics import accuracy_score


def main():
    api = MongoDb()
    laebl_1 = api.get_record(
        collection_name=COLLECTION_NAME,
        collection_id=COLLECTION_ID,
        label="SPORTS",
        limit=100,
    )

    laebl_2 = api.get_record(
        collection_name=COLLECTION_NAME,
        collection_id=COLLECTION_ID,
        label="COMEDY",
        limit=100,
    )

    # laebl_1 = api.get_record(
    #     collection_name=COLLECTION_NAME,
    #     collection_id=COLLECTION_ID,
    #     label=0,
    #     limit=10,
    # )
    # laebl_2 = api.get_record(
    #     collection_name=COLLECTION_NAME,
    #     collection_id=COLLECTION_ID,
    #     label=4,
    #     limit=10,
    # )

    instances = laebl_1 + laebl_2
    labels = [sample["record"]["label"] for sample in laebl_1 + laebl_2]

    instances, labels = shuffle(instances, labels, random_state=0)

    Transformator = transformator()
    Transformator.fit(instances, labels)
    # Transformator.fit(labels)

    peony_model = PeonyBoxModel(
        Transformator,
        active_learning_step=10,
        acquisition_function=hac_sampling,  # entropy_sampling, batch_bald,
    )
    peony_model.bayesian_dropout_nn.fit(instances[:50], labels[:50])
    # peony_model.bayesian_denfi_nn.reset()
    peony_model.bayesian_dropout_nn.epsilon_greedy_coef = 1
    indexes = peony_model.bayesian_dropout_nn.get_learning_samples(instances[50:])

    add_training = [instances[index] for index in indexes.tolist()]
    add_labels = [labels[index] for index in indexes.tolist()]

    peony_model.bayesian_dropout_nn.add_new_learning_samples(add_training, add_labels)
    peony_model.bayesian_dropout_nn.fit(instances, labels)

    start_time = time.time()
    k_fold = k_fold_corss_validation(peony_model.bayesian_dropout_nn, Transformator, instances, labels, 2)
    print(f"elapsed time is {time.time() - start_time}")

    print(auc_metrics(k_fold))

    scores = [accuracy_score(eval["true"], eval["predicted"], normalize=True) for eval in k_fold]

    print(scores)
    print("test")


if __name__ == "__main__":
    main()
