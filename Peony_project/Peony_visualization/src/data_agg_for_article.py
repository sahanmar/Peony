import pandas as pd
import numpy as np

from PeonyPackage.PeonyDb import MongoDb

from typing import Dict

dev_func = np.std
mean_func = np.mean
api = MongoDb()


def create_table_str(res):
    return f"{round(mean_func(res, axis=0)[-1][0],3)}$\pm${round(dev_func(res, axis=0)[-1][0],3)}"


def create_evolution_row(res):
    return [
        f"{round(mean_func(res, axis=0)[0][0],3)}$\pm${round(dev_func(res, axis=0)[0][0],3)}",
        f"{round(mean_func(res, axis=0)[49][0],3)}$\pm${round(dev_func(res, axis=0)[49][0],3)}",
        f"{round(mean_func(res, axis=0)[99][0],3)}$\pm${round(dev_func(res, axis=0)[99][0],3)}",
        f"{round(mean_func(res, axis=0)[149][0],3)}$\pm${round(dev_func(res, axis=0)[149][0],3)}",
        f"{round(mean_func(res, axis=0)[199][0],3)}$\pm${round(dev_func(res, axis=0)[199][0],3)}",
    ]


def get_res_from_db(alg, acsq_func, category):
    return api.get_model_results(
        {"model": alg, "acquisition_function": acsq_func, "category_1": category,}
    )[0]["results"]


def create_non_nn_table():
    category = "SPORTS"

    # Random
    svm_tfidf_ran = api.get_model_results(
        {"model": "svm", "acquisition_function": "random", "category_1": category}
    )[0]["results"]
    svm_fasttext_ran = api.get_model_results(
        {
            "model": "svm_fast_text_embeddings",
            "acquisition_function": "random",
            "category_1": category,
        }
    )[0]["results"]
    rf_tfidf_ran = api.get_model_results(
        {
            "model": "random_forest",
            "acquisition_function": "random",
            "category_1": category,
        }
    )[0]["results"]
    rf_fasttext_ran = api.get_model_results(
        {
            "model": "random_forest_fast_text_embeddings",
            "acquisition_function": "random",
            "category_1": category,
        }
    )[0]["results"]
    # Entropy
    svm_tfidf_ent = api.get_model_results(
        {"model": "svm", "acquisition_function": "entropy", "category_1": category}
    )[0]["results"]
    svm_fasttext_ent = api.get_model_results(
        {
            "model": "svm_fast_text_embeddings",
            "acquisition_function": "entropy",
            "category_1": category,
        }
    )[0]["results"]
    rf_tfidf_ent = api.get_model_results(
        {
            "model": "random_forest",
            "acquisition_function": "entropy",
            "category_1": category,
        }
    )[0]["results"]
    rf_fasttext_ent = api.get_model_results(
        {
            "model": "random_forest_fast_text_embeddings",
            "acquisition_function": "entropy",
            "category_1": category,
        }
    )[0]["results"]

    res = [
        [
            create_table_str(svm_tfidf_ran),
            create_table_str(svm_tfidf_ent),
            create_table_str(svm_fasttext_ran),
            create_table_str(svm_fasttext_ent),
        ],
        [
            create_table_str(rf_tfidf_ran),
            create_table_str(rf_tfidf_ent),
            create_table_str(rf_fasttext_ran),
            create_table_str(rf_fasttext_ent),
        ],
    ]

    df = pd.DataFrame(res)
    df.columns = [
        "TF-INF Random",
        "TF-INF Entropy",
        "Fast Text Random",
        "Fast Text Entopy",
    ]
    df["Algorithms"] = ["SVM", "Random Forest"]

    return df.set_index("Algorithms")


def create_nn_table():
    categories = [
        "CRIME",
        "SPORTS",
        "POLITICS",
        "TECH",
        "COLLEGE",
        "POSITIVE_EMOTIONS_TWEETS",
    ]
    acq_funcs = ["entropy"]  # "entropy"]
    algs = [
        "bayesian_sgld_nn_fast_text_embeddings",
        "bayesian_denfi_nn_hot_start_fast_text_embeddings",
        "bayesian_denfi_v_2_0.3_fast_text_embeddings",
        "bayesian_dropout_nn_fast_text_embeddings",
        "bayesian_dropout_nn_hot_start_fast_text_embeddings",
        "bayesian_dropout_hot_start_w_noise_0.3_fast_text_embeddings",
    ]
    dev_func = np.std
    mean_func = np.mean

    list_w_results = [
        [
            create_table_str(get_res_from_db(alg, acq_func, category))
            for category in categories
            for acq_func in acq_funcs
        ]
        for alg in algs
    ]

    df = pd.DataFrame(list_w_results)
    df.columns = [
        # "CRIME Random",
        "CRIME Entropy",
        # "SPORTS Random",
        "SPORTS Entopy",
        # "POLITICS Random",
        "POLITICS Entopy",
        # "TECH Random",
        "TECH Entopy",
        # "EDUCATION Random",
        "EDUCATION Entopy",
        # TWEETS Random
        "TWEETS Entropy",
    ]

    df["Algorithms"] = [
        "SGLD",
        "DENFI V1",
        "DENFI V2",
        "Dropout cold start",
        "Dropout hot start",
        "Dropout hot start w noise",
    ]

    return df.set_index("Algorithms")


def create_evloution_table():
    category = "TECH"
    models = [0.1, 0.2, 0.3, 0.4, 0.6, 1]
    dev_func = np.std
    mean_func = np.mean

    list_w_results = [
        create_evolution_row(get_res_from_db(alg, "entropy", category))
        for alg in [
            f"bayesian_dropout_hot_start_w_noise_{model}_fast_text_embeddings"
            # f"bayesian_denfi_v_2_{model}_fast_text_embeddings"
            for model in models
        ]
    ]

    df = pd.DataFrame(list_w_results)
    df.columns = [
        "0",
        "50",
        "100",
        "150",
        "200",
    ]

    df["Noise Variance"] = ["0.1", "0.2", "0.3", "0.4", "0.6", "1"]

    return df.set_index("Noise Variance")
