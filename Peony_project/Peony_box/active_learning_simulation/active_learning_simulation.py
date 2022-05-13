from telnetlib import IP
from PeonyPackage.PeonyDb import MongoDb
from Peony_box.active_learning_simulation.utils import active_learning_simulation

from Peony_box.src.transformators.HuffPost_transformator import (
    RoBERTaWordEmbeddings as transformator,
)

# from Peony_database.src.datasets.Tweets_emotions_dataset import COLLECTION_NAME, COLLECTION_ID
from Peony_database.src.datasets.fake_news_detection import COLLECTION_NAME, COLLECTION_ID
# from Peony_database.src.datasets.gibberish import COLLECTION_NAME, COLLECTION_ID
# from Peony_database.src.datasets.amazon_reviews import COLLECTION_NAME, COLLECTION_ID

from Peony_box.src.acquisition_functions.functions import (
    entropy_sampling,
    batch_bald,
    hac_entropy_sampling,
    hac_bald_sampling,
    power_bald,
    bald_sampling,
)
from Peony_visualization.src.peony_visualization import visualize_two_auc_evolutions

from sklearn.utils import shuffle
import numpy as np


ASC_FUNC_MAP = {
    "random": None,
    "batch_bald": batch_bald,
    "power_bald": power_bald,
    "entropy_sampling": entropy_sampling,
    "bald_sampling": bald_sampling,
    "hac_entropy_sampling": hac_entropy_sampling,
    "hac_bald_sampling": hac_bald_sampling,
}

ALGORITHM_1 = "bayesian_dropout"
ALGORITHM_2 = "nn"

ENCODER = ""
NOISE = ""

ASC_FUNC_1 = "hac_entropy_sampling"
ASC_FUNC_2 = "hac_bald_sampling"
ASC_FUNC_3 = "random"
ASC_FUNC_4 = "entropy_sampling"
ASC_FUNC_5 = "bald_sampling"

LABELS = [0,1]#[0, 1]#[0,4] #[3, 5]

LIMIT_1 = 2000
LIMIT_2 = 2000

ACTIVE_LEARNING_LOOPS = 5
ACTIVE_LEARNING_STEP = 50
ACTIVE_LEARNING_SAMPLES = 20
INITIAL_TRAINING_DATA_SIZE = 10

MODEL_1 = f"{ALGORITHM_1}_{ENCODER}_{NOISE}_{ASC_FUNC_1}_{ACTIVE_LEARNING_STEP}_{ACTIVE_LEARNING_SAMPLES}"
MODEL_2 = f"{ALGORITHM_2}_{ENCODER}_{NOISE}_{ASC_FUNC_2}_{ACTIVE_LEARNING_STEP}_{ACTIVE_LEARNING_SAMPLES}"

CATEGORY_1 = "Fake_news_detection_0" #"Tweet_emotion_0" #"GIBBERISH 1"
CATEGORY_2 = "Fake_news_detection_1" #"Tweet_emotion_4" #"GIBBERISH 2"


def main():

    api = MongoDb()

    records_1 = api.get_record(
        collection_name=COLLECTION_NAME,
        collection_id=COLLECTION_ID,
        label=LABELS[0],
        limit=LIMIT_1,
    )
    records_2 = api.get_record(
        collection_name=COLLECTION_NAME,
        collection_id=COLLECTION_ID,
        label=LABELS[1],
        limit=LIMIT_2,
    )

    # Define model specifications
    transformation_needed = False

    instances = records_1 + records_2
    labels = [sample["record"]["label"] for sample in records_1 + records_2]

    instances_from_db, labels_from_db = shuffle(instances, labels, random_state=0)

    # HuffPostTransform = word_embed_transformator()

    HuffPostTransform = (
        transformator()
    )  # I'm using here not HuffPost transformator but I'm too lazy to change all variable names

    HuffPostTransform.fit(labels_from_db)

    if transformation_needed:
        instances = instances_from_db
        labels = labels_from_db
    else:
        instances = HuffPostTransform.transform_instances(instances_from_db)
        labels = HuffPostTransform.transform_labels(labels_from_db)

    # Get AUC results from an active learning simulation

    auc_learning_0 = active_learning_simulation(
        HuffPostTransform,
        ASC_FUNC_MAP[ASC_FUNC_1],
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_SAMPLES,
        ACTIVE_LEARNING_STEP,
        ALGORITHM_2,
        instances,
        labels,
        INITIAL_TRAINING_DATA_SIZE,
        transformation_needed,
    )

    list_to_upload = [
        ALGORITHM_2,
        ASC_FUNC_1,
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_STEP,
        ACTIVE_LEARNING_SAMPLES,
        INITIAL_TRAINING_DATA_SIZE,
        LIMIT_1 + LIMIT_2 - INITIAL_TRAINING_DATA_SIZE - ACTIVE_LEARNING_SAMPLES * ACTIVE_LEARNING_STEP,
        CATEGORY_1,
        CATEGORY_2,
        auc_learning_0,
    ]

    api.load_model_results(*list_to_upload)

    print("Zeros simulation is ready...")
    
    auc_learning_1 = active_learning_simulation(
        HuffPostTransform,
        ASC_FUNC_MAP[ASC_FUNC_1],
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_SAMPLES,
        ACTIVE_LEARNING_STEP,
        ALGORITHM_1,
        instances,
        labels,
        INITIAL_TRAINING_DATA_SIZE,
        transformation_needed,
    )

    list_to_upload = [
        f"{ALGORITHM_1}_d_0_2",#_1_ens",
        ASC_FUNC_1,
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_STEP,
        ACTIVE_LEARNING_SAMPLES,
        INITIAL_TRAINING_DATA_SIZE,
        LIMIT_1 + LIMIT_2 - INITIAL_TRAINING_DATA_SIZE - ACTIVE_LEARNING_SAMPLES * ACTIVE_LEARNING_STEP,
        CATEGORY_1,
        CATEGORY_2,
        auc_learning_1,
    ]

    api.load_model_results(*list_to_upload)

    print("First simulation is ready...")

    auc_learning_2 = active_learning_simulation(
        HuffPostTransform,
        ASC_FUNC_MAP[ASC_FUNC_2],
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_SAMPLES,
        ACTIVE_LEARNING_STEP,
        ALGORITHM_1,
        instances,
        labels,
        INITIAL_TRAINING_DATA_SIZE,
        transformation_needed,
    )

    list_to_upload = [
        f"{ALGORITHM_1}_d_0_2",#_1_ens",
        ASC_FUNC_2,
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_STEP,
        ACTIVE_LEARNING_SAMPLES,
        INITIAL_TRAINING_DATA_SIZE,
        LIMIT_1 + LIMIT_2 - INITIAL_TRAINING_DATA_SIZE - ACTIVE_LEARNING_SAMPLES * ACTIVE_LEARNING_STEP,
        CATEGORY_1,
        CATEGORY_2,
        auc_learning_2,
    ]

    api.load_model_results(*list_to_upload)

    print("Second simulation is ready...")

    auc_learning_3 = active_learning_simulation(
        HuffPostTransform,
        ASC_FUNC_MAP[ASC_FUNC_3],
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_SAMPLES,
        ACTIVE_LEARNING_STEP,
        ALGORITHM_1,
        instances,
        labels,
        INITIAL_TRAINING_DATA_SIZE,
        transformation_needed,
    )

    list_to_upload = [
        f"{ALGORITHM_1}_d_0_2",#_1_ens",
        ASC_FUNC_3,
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_STEP,
        ACTIVE_LEARNING_SAMPLES,
        INITIAL_TRAINING_DATA_SIZE,
        LIMIT_1 + LIMIT_2 - INITIAL_TRAINING_DATA_SIZE - ACTIVE_LEARNING_SAMPLES * ACTIVE_LEARNING_STEP,
        CATEGORY_1,
        CATEGORY_2,
        auc_learning_3,
    ]

    api.load_model_results(*list_to_upload)

    print("Third simulation is ready...")

    auc_learning_4 = active_learning_simulation(
        HuffPostTransform,
        ASC_FUNC_MAP[ASC_FUNC_4],
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_SAMPLES,
        ACTIVE_LEARNING_STEP,
        ALGORITHM_1,
        instances,
        labels,
        INITIAL_TRAINING_DATA_SIZE,
        transformation_needed,
    )

    list_to_upload = [
        f"{ALGORITHM_1}_d_0_2",#_1_ens",
        ASC_FUNC_4,
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_STEP,
        ACTIVE_LEARNING_SAMPLES,
        INITIAL_TRAINING_DATA_SIZE,
        LIMIT_1 + LIMIT_2 - INITIAL_TRAINING_DATA_SIZE - ACTIVE_LEARNING_SAMPLES * ACTIVE_LEARNING_STEP,
        CATEGORY_1,
        CATEGORY_2,
        auc_learning_4,
    ]

    api.load_model_results(*list_to_upload)

    print("Fourth simulation is ready...")

    auc_learning_5 = active_learning_simulation(
        HuffPostTransform,
        ASC_FUNC_MAP[ASC_FUNC_4],
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_SAMPLES,
        ACTIVE_LEARNING_STEP,
        ALGORITHM_1,
        instances,
        labels,
        INITIAL_TRAINING_DATA_SIZE,
        transformation_needed,
    )

    list_to_upload = [
        f"{ALGORITHM_1}_d_0_2",#_1_ens",
        ASC_FUNC_5,
        ACTIVE_LEARNING_LOOPS,
        ACTIVE_LEARNING_STEP,
        ACTIVE_LEARNING_SAMPLES,
        INITIAL_TRAINING_DATA_SIZE,
        LIMIT_1 + LIMIT_2 - INITIAL_TRAINING_DATA_SIZE - ACTIVE_LEARNING_SAMPLES * ACTIVE_LEARNING_STEP,
        CATEGORY_1,
        CATEGORY_2,
        auc_learning_5,
    ]

    api.load_model_results(*list_to_upload)

    print("Fifth simulation is ready...")

    #visualize_two_auc_evolutions(auc_learning_1, auc_learning_2)

if __name__ == "__main__":
    main()
