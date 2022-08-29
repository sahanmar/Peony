import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import re
import pandas as pd

from typing import List, Any, Dict
from Peony_visualization.src.batch_active_learning_article.result_ids import DATA
from matplotlib.ticker import FormatStrFormatter


def visualize_auc_evolutions(
    ax,
    markevery,
    auc_seq_passive_1,
    auc_seq_active_1,
    auc_seq_active_2,
    model_1,
    model_2,
    batch,
    title,
    index,
):

    auc_1_passive_mean = auc_seq_passive_1["results_mean"].values.flatten().tolist()[0]
    auc_1_passive_std = auc_seq_passive_1["results_std"].values.flatten().tolist()[0]

    auc_1_active_mean = auc_seq_active_1["results_mean"].values.flatten().tolist()[0]
    auc_1_active_std = auc_seq_active_1["results_std"].values.flatten().tolist()[0]

    auc_2_active_mean = auc_seq_active_2["results_mean"].values.flatten().tolist()[0]
    auc_2_active_std = auc_seq_active_2["results_std"].values.flatten().tolist()[0]

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.2)
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_1_passive_mean,
        linestyle="--",
        marker="+",
        markevery=markevery,
        color="b",
        lw=1,
        label=f"Random Selection {model_1} mean",
        alpha=0.4,
    )
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_1_passive_mean + auc_1_passive_std,
        linestyle="-",
        color="b",
        alpha=0.1,
    )
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_1_passive_mean - auc_1_passive_std,
        linestyle="-",
        color="b",
        alpha=0.1,
    )

    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_1_active_mean,
        linestyle="-",
        marker="+",
        markevery=markevery,
        color="b",
        lw=1,
        label=f"Active Learning {model_1} mean",
    )
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_1_active_mean + auc_1_active_std,
        linestyle="-",
        color="b",
        alpha=0.1,
    )
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_1_active_mean - auc_1_active_std,
        linestyle="-",
        color="b",
        alpha=0.1,
    )

    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_2_active_mean,
        linestyle="-",
        marker="*",
        markevery=markevery,
        color="g",
        lw=1,
        label=f"Active Learning {model_2} mean",
    )
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_2_active_mean + auc_2_active_std,
        linestyle="-",
        color="g",
        alpha=0.1,
    )
    ax.plot(
        [i for i in range(batch, 1000 + batch, batch)],
        auc_2_active_mean - auc_2_active_std,
        linestyle="-",
        color="g",
        alpha=0.1,
    )

    ax.fill_between(
        [i for i in range(batch, 1000 + batch, batch)],
        (auc_1_passive_mean + auc_1_passive_std),
        (auc_1_passive_mean - auc_1_passive_std),
        alpha=0.05,
        color="b",
    )

    ax.fill_between(
        [i for i in range(batch, 1000 + batch, batch)],
        (auc_1_active_mean + auc_1_active_std),
        (auc_1_active_mean - auc_1_active_std),
        alpha=0.05,
        color="b",
    )

    ax.fill_between(
        [i for i in range(batch, 1000 + batch, batch)],
        (auc_2_active_mean + auc_2_active_std),
        (auc_2_active_mean - auc_2_active_std),
        alpha=0.05,
        color="g",
    )

    ax.set_xlabel("Requests", fontsize=13.5)
    if index == 0:
        ax.set_ylabel("AUC", fontsize=13.5)
    ax.set_title(
        title,
        fontsize=16,
    )
    ax.legend(loc="lower right", fontsize=10)

    return ax


def get_collection_results() -> Dict[str, List[List[float]]]:
    collection_results = {}
    with open("Peony_visualization/src/batch_active_learning_article/collection.json", "r") as f:
        for l in f.readlines():
            data = json.loads(l)
            collection_results[data["_id"]["$oid"]] = data["results"]
    return collection_results


def merge_resuls_and_metadata(
    resutls: Dict[str, List[List[float]]], metadata: List[List[str]]
) -> pd.DataFrame:
    list_2_df: List[Any] = []
    for row in metadata:
        id_res = np.array(resutls[row[0]]).squeeze(2)

        batch = re.search(r"(100|50|20|10)", row[2])
        batch_span = batch.span()  # type: ignore
        batch_int = int(batch.group(0))  # type: ignore

        warm_start = re.search(r"warm_start", row[2])
        if warm_start:
            alg = row[2][: batch_span[0] - 1] + "_warm_start_" + row[3]
        else:
            alg = row[2][: batch_span[0] - 1] + "_" + row[3]
        mean = np.mean(id_res, axis=0)
        std = np.std(id_res, axis=0)
        list_2_df.append(row[0:2] + [alg, batch_int, id_res, mean, std])
    header = ["id", "dataset", "algorithm", "batch", "results", "results_mean", "results_std"]
    df = pd.DataFrame(list_2_df)
    df.columns = header
    return df


def main():

    collection_results = get_collection_results()
    df = merge_resuls_and_metadata(collection_results, DATA)
    df = df[
        (df["algorithm"] != "nn_warm_start_hac_bald")
        & (df["algorithm"] != "nn_warm_start_bald")
        & (df["algorithm"] != "denfi_hac_bald")
        & (df["algorithm"] != "denfi_bald")
    ]

    algorithms = [
        (
            "mc_dropout_entropy",
            "mc_dropout_random",
            "nn_min_margin",
            10,
            "Fake news detection",
            "MC Dropout\nHAC Entropy",
            "HAC\nMin-margin",
            "Fake News Detection",
        ),
        (
            "denfi_entropy",
            "denfi_random",
            "nn_min_margin",
            20,
            "Amazon Review 3, 5",
            "DEnFi\nEntropy",
            "HAC\nMin-margin",
            "Amazon Reviews 3, 5",
        ),
        (
            "nn_warm_start_entropy",
            "nn_warm_start_random",
            "nn_min_margin",
            50,
            "Gibberish",
            "NN Entropy\nWarm-start",
            "HAC\nMin-margin",
            "Gibberish",
        ),
        (
            "nn_warm_start_entropy",
            "nn_warm_start_random",
            "nn_min_margin",
            100,
            "Tweet_emotion_detection",
            "NN Entropy\nWarm-start",
            "HAC\nMin-margin",
            "Twitter Sentiment",
        ),
    ]

    for index, (q_a_1, q_r_1, q_a_2, batch, q_data, alg_legend_1, alg_legend_2, title_category) in enumerate(
        algorithms
    ):

        al_1 = df[(df["algorithm"] == q_a_1) & (df["batch"] == batch) & (df["dataset"] == q_data)]
        al_2 = df[(df["algorithm"] == q_a_2) & (df["batch"] == batch) & (df["dataset"] == q_data)]

        rs_1 = df[(df["algorithm"] == q_r_1) & (df["batch"] == batch) & (df["dataset"] == q_data)]

        ax = plt.subplot(1, 4, index + 1)

        visualize_auc_evolutions(
            ax,
            batch,
            rs_1,
            al_1,
            al_2,
            alg_legend_1,
            alg_legend_2,
            batch,
            f"{title_category}, batch {batch}",
            index,
        )

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
