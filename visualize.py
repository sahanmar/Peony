from PeonyPackage.PeonyDb import MongoDb
from Peony_visualization.src.peony_visualization import visualize_two_auc_evolutions

api = MongoDb()

# Random acquisition function
svm_random_sampling_results = api.get_model_results(
    {
        "model": "bayesian_denfi_nn_hot_start_fast_text_embeddings",
        "acquisition_function": "random",
        "category_1": "POSITIVE_EMOTIONS_TWEETS",
    }
)
svm_random_sampling_results = [
    item for val in svm_random_sampling_results for item in val["results"]
]

# Entropy acquisition function
svm_false_positive_sampling_results = api.get_model_results(
    {
        "model": "bayesian_denfi_nn_hot_start_fast_text_embeddings",
        "acquisition_function": "entropy",
        "category_1": "POSITIVE_EMOTIONS_TWEETS",
    }
)
svm_false_positive_sampling_results = [
    item for val in svm_false_positive_sampling_results for item in val["results"]
]

# we use zero index because database returns list even if it is only one element in list

visualize_two_auc_evolutions(
    svm_random_sampling_results, svm_false_positive_sampling_results
)
