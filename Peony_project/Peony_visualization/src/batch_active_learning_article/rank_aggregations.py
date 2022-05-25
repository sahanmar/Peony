import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math

from itertools import chain
from Peony_visualization.src.batch_active_learning_article.result_ids import DATA
from typing import Any, Dict, List, Optional
from matplotlib.ticker import FormatStrFormatter


def plot_batch_evolutions(dfs: List[pd.DataFrame]) -> None:

    results = {}
    batches = list(reversed([int(df.columns[1].split("_")[0]) for df in dfs]))
    markevery = 1

    algos_2_plot = ["nn_warm_start_entropy", "nn_warm_start_bald", "nn_min_margin", "mc_dropout_entropy"]

    for df in dfs:
        for _, (algo, algo_mean, algo_std) in df.iterrows():
            if algo in algos_2_plot:
                results.setdefault(algo, {"mean": [], "std": []})
                results[algo]["mean"].append(algo_mean)
                results[algo]["std"].append(algo_std)

    for algo, stats in results.items():
        stats["mean"] = list(reversed(stats["mean"]))
        stats["std"] = list(reversed(stats["std"]))
        plt.grid(alpha=0.2)
        plt.plot(
            batches,
            stats["mean"],
            linestyle="--",
            marker="+",
            markevery=markevery,
            color="b",
            lw=1,
            label=algo,
            alpha=0.4,
        )
        upper_bound = [m + s for m, s in zip(stats["mean"], stats["std"])]
        lower_bound = [m - s for m, s in zip(stats["mean"], stats["std"])]
        plt.plot(batches, upper_bound, linestyle="-", color="b", alpha=0.1)
        plt.plot(batches, lower_bound, linestyle="-", color="b", alpha=0.1)

        plt.fill_between(
            batches,
            upper_bound,
            lower_bound,
            alpha=0.05,
            color="b",
        )

    plt.show()


def heatmap_batch(
    dfs: List[pd.DataFrame],
    alg_title: Optional[str] = None,
    subplot: Optional[Any] = None,
    label_size: int = 8,
) -> None:
    df = dfs[0].merge(dfs[1], how="left").merge(dfs[2], how="left").merge(dfs[3], how="left")
    df_h = df[["10_ranks_mean", "20_ranks_mean", "50_ranks_mean", "100_ranks_mean"]]

    # fig, ax = plt.subplots(figsize=(8, 6))
    if subplot is not None:
        sns.set(font_scale=1.12)
        ax = sns.heatmap(
            df_h,
            cmap="Blues_r",
            linewidths=0.0,
            annot=True,
            xticklabels=["10\n ", "20\n ", "50\n ", "100\n "],
            yticklabels=[
                "HAC\nMin-margin",
                "MC Dropout\nHAC Entropy",
                "MC Dropout\nHAC BALD",
                "MC Dropout\nEntropy",
                "MC Dropout\nBALD",
                "MC Dropout\nRandom",
                "NN HAC Entropy\nWarm-start",
                # "NN HAC BALD\nWarm-start",
                "NN Entropy\nWarm-start",
                # "NN BALD\nWarm-start",
                "NN Random\nWarm-start",
            ],
            vmin=math.floor(df_h.min().min()),
            vmax=math.ceil(df_h.max().max()),
            ax=subplot,
            cbar=False,
        )
        ax.tick_params(labelsize=label_size)
        ax.set_title(
            "Aggregated batch size mean rank through all algorithms" if not alg_title else alg_title,
            pad=20,
            fontsize=label_size + 6,
        )
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=40)
    else:
        ax = sns.heatmap(
            df_h,
            cmap="Blues_r",
            linewidths=0.0,
            annot=True,
            xticklabels=["10\n ", "20\n ", "50\n ", "100\n "],
            yticklabels=[
                "HAC\nMin-margin",
                "MC Dropout\nHAC Entropy",
                "MC Dropout\nHAC BALD",
                "MC Dropout\nEntropy",
                "MC Dropout\nBALD",
                "MC Dropout\nRandom",
                "NN HAC Entropy\nWarm-start",
                # "NN HAC BALD\nWarm-start",
                "NN Entropy\nWarm-start",
                # "NN BALD\nWarm-start",
                "NN Random\nWarm-start",
            ],
            vmin=math.floor(df_h.min().min()),
            vmax=math.ceil(df_h.max().max()),
        )
        ax.tick_params(labelsize=label_size)
        ax.set_title(
            f"Aggregated batch size mean rank through {'all algorithms' if not alg_title else alg_title}",
            pad=20,
        )
        plt.tight_layout()
        plt.show()


def heatmap_datasets(dfs: List[pd.DataFrame]) -> None:

    if len(dfs) == 3:
        df = dfs[0].merge(dfs[1], how="left").merge(dfs[2], how="left")

        df_h = df[
            [
                "Amazon Review 3, 5_ranks_mean",
                "Fake news detection_ranks_mean",
                "Tweet_emotion_detection_ranks_mean",
            ]
        ]

        xlabels = [
            "Amazon Reviews\n3, 5",
            "Fake News\nDetection",
            "Twitter\nSentiment",
        ]

        ylabels = [
            "HAC\nMin-margin",
            "MC Dropout\nHAC Entropy",
            "MC Dropout\nHAC BALD",
            "MC Dropout\nEntropy",
            "MC Dropout\nBALD",
            "MC Dropout\nRandom",
            "DEnFi\nHAC Entropy",
            "DEnFi\nHAC BALD",
            "DEnFi\nEntropy",
            "DEnFi\nBALD",
            "DEnFi\nRandom",
            "NN HAC Entropy\nWarm-start",
            # "NN HAC BALD\nWarm-start",
            "NN Entropy\nWarm-start",
            # "NN BALD\nWarm-start",
            "NN Random\nWarm-start",
        ]

        # fig, ax = plt.subplots(figsize=(8, 6))
        sns.set(font_scale=1)
        ax = sns.heatmap(
            df_h,
            cmap="Blues_r",
            linewidths=0.0,
            annot=True,
            xticklabels=xlabels,
            yticklabels=ylabels,
            vmin=math.floor(df_h.min().min()),
            vmax=math.ceil(df_h.max().max()),
            # cbar=False if index == 1 else True,
        )
        ax.tick_params(labelsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
        # ax.set_title("Aggregated mean rank given datasets and algorithm", pad=20)
        plt.tight_layout()
        plt.show()

    else:
        df = (
            dfs[0]
            .merge(dfs[1], how="left")
            .merge(dfs[2], how="left")
            .merge(dfs[3], how="left")
            .merge(dfs[4], how="left")
        )

        df_h = df[
            [
                "Amazon Review 1, 5_ranks_mean",
                "Amazon Review 3, 5_ranks_mean",
                "Gibberish_ranks_mean",
                "Fake news detection_ranks_mean",
                "Tweet_emotion_detection_ranks_mean",
            ]
        ]

        xlabels = [
            "Amazon\nReviews 1, 5",
            "Amazon\nReviews 3, 5",
            "Gibberish",
            "Fake News\nDetection",
            "Twitter\nSentiment",
        ]
        ylabels = [
            "HAC\nMin-margin",
            "MC Dropout\nHAC Entropy",
            "MC Dropout\nHAC BALD",
            "MC Dropout\nEntropy",
            "MC Dropout\nBALD",
            "MC Dropout\nRandom",
            "NN HAC Entropy\nWarm-start",
            # "NN HAC BALD\nWarm-start",
            "NN Entropy\nWarm-start",
            # "NN BALD\nWarm-start",
            "NN Random\nWarm-start",
        ]

        # fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.heatmap(
            df_h,
            cmap="Blues_r",
            linewidths=0.0,
            annot=True,
            xticklabels=xlabels,
            yticklabels=ylabels,
            vmin=math.floor(df_h.min().min()),
            vmax=math.ceil(df_h.max().max()),
        )
        ax.tick_params(labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
        # ax.set_title("Aggregated mean rank given datasets and algorithm", pad=20)
        plt.tight_layout()
        plt.show()


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


def get_ranks(df: pd.DataFrame) -> pd.DataFrame:
    results = df["results_mean"].to_list()
    sorted_results = []
    for v in zip(*results):
        res = list(zip(*sorted(zip(*[v, list(range(1, len(v) + 1))]), key=lambda x: x[0], reverse=True)))[1]

        sorted_results.append(
            list(zip(*sorted(zip(*[res, list(range(1, len(v) + 1))]), key=lambda x: x[0])))[1]
        )
    transposed_sorted_results = list(zip(*sorted_results))

    df = pd.DataFrame(list(zip(df["algorithm"].to_list(), transposed_sorted_results)))
    df.columns = ["algorithm", "ranks"]
    return df


def aggregate_ranks_through_batches(ranks: List[List[float]], std=False) -> List[float]:

    results = []
    for res_list in ranks:
        step = len(res_list) // 10
        for i in range(0, len(res_list), step):
            results.append(res_list[i])

    return np.mean(results) if std == False else np.std(results)


def get_batch_rank(df: pd.DataFrame) -> List[pd.DataFrame]:
    df_100 = df[df["batch"] == 100]
    df_50 = df[df["batch"] == 50]
    df_20 = df[df["batch"] == 20]
    df_10 = df[df["batch"] == 10]

    def group_by(df: pd.DataFrame, batch_size: str) -> pd.DataFrame:
        df_by_datasets = pd.concat(
            [get_ranks(df[df["dataset"] == dataset]) for dataset in set(df["dataset"].to_list())]
        )
        df_mean = (
            df_by_datasets.groupby("algorithm", sort=False)
            .apply(lambda x: aggregate_ranks_through_batches(list(x["ranks"])))
            .reset_index()
        )
        df_mean.columns = ["algorithm", f"{batch_size}_ranks_mean"]
        df_std = (
            df_by_datasets.groupby("algorithm", sort=False)
            .apply(lambda x: aggregate_ranks_through_batches(list(x["ranks"]), True))
            .reset_index()
        )
        df_std.columns = ["algorithm", f"{batch_size}_ranks_std"]
        df = df_mean[["algorithm", f"{batch_size}_ranks_mean"]]
        df[f"{batch_size}_ranks_std"] = df_std[f"{batch_size}_ranks_std"]
        return df.round(3)

    return [group_by(df_100, "100"), group_by(df_50, "50"), group_by(df_20, "20"), group_by(df_10, "10")]


def get_dataset_rank(df: pd.DataFrame) -> List[pd.DataFrame]:
    dfs = [df[df["dataset"] == dataset] for dataset in set(df["dataset"].to_list())]

    def group_by(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        df_by_datasets = pd.concat(
            [get_ranks(df[df["batch"] == batch]) for batch in set(df["batch"].to_list())]
        )
        df_mean = (
            df_by_datasets.groupby("algorithm", sort=False)
            .apply(lambda x: aggregate_ranks_through_batches(list(x["ranks"])))
            .reset_index()
        )
        df_mean.columns = ["algorithm", f"{dataset}_ranks_mean"]
        df_std = (
            df_by_datasets.groupby("algorithm", sort=False)
            .apply(lambda x: aggregate_ranks_through_batches(list(x["ranks"]), True))
            .reset_index()
        )
        df_std.columns = ["algorithm", f"{dataset}_ranks_std"]
        df = df_mean[["algorithm", f"{dataset}_ranks_mean"]]
        df[f"{dataset}_ranks_std"] = df_std[f"{dataset}_ranks_std"]
        return df.round(3)

    return [group_by(dataframe, dataset) for dataframe, dataset in zip(dfs, set(df["dataset"].to_list()))]


def get_batch_rank_subplots(df: pd.DataFrame) -> None:
    label_size = 13.5
    heatmap_batch(get_batch_rank(df), "All Datasets", plt.subplot(2, 3, 1), label_size)
    heatmap_batch(
        get_batch_rank(df[df["dataset"] == "Tweet_emotion_detection"]),
        "Twitter Sentiment",
        plt.subplot(2, 3, 2),
        label_size,
    )
    heatmap_batch(
        get_batch_rank(df[df["dataset"] == "Gibberish"]), "Gibberish", plt.subplot(2, 3, 3), label_size
    )
    heatmap_batch(
        get_batch_rank(df[df["dataset"] == "Amazon Review 3, 5"]),
        "Amazon Reviews 3, 5",
        plt.subplot(2, 3, 4),
        label_size,
    )
    heatmap_batch(
        get_batch_rank(df[df["dataset"] == "Amazon Review 1, 5"]),
        "Amazon Reviews 1, 5",
        plt.subplot(2, 3, 5),
        label_size,
    )
    heatmap_batch(
        get_batch_rank(df[df["dataset"] == "Fake news detection"]),
        "Fake News Detection",
        plt.subplot(2, 3, 6),
        label_size,
    )
    plt.tight_layout()
    plt.show()


def get_batch_rank_subplots_denfi(df: pd.DataFrame) -> None:
    df_denfi = df[
        (df["batch"] == 20) & (df["dataset"] != "Gibberish") & (df["dataset"] != "Amazon Review 1, 5")
    ]
    dataset_ranks = get_dataset_rank(df_denfi)
    heatmap_datasets(dataset_ranks)
    # algos = [
    #     "nn_min_margin",
    #     "denfi_hac_entropy",
    #     "denfi_hac_bald",
    #     "denfi_entropy",
    #     "denfi_bald",
    #     "denfi_random",
    #     "mc_dropout_random",
    # ]
    # first_subplot = [d_r[d_r["algorithm"].isin(algos)] for d_r in dataset_ranks]
    # second_subplot = [d_r[~d_r["algorithm"].isin(algos)] for d_r in dataset_ranks]

    # label_size = 13.5

    # df = dataset_ranks[0].merge(dataset_ranks[1], how="left").merge(dataset_ranks[2], how="left")

    # df_h = df[
    #     [
    #         "Amazon Review 3, 5_ranks_mean",
    #         "Fake news detection_ranks_mean",
    #         "Tweet_emotion_detection_ranks_mean",
    #     ]
    # ]

    # min_v = math.floor(df_h.min().min())
    # max_v = math.floor(df_h.max().max())
    # heatmap_datasets(first_subplot, 1, plt.subplot(1, 2, 1), min_v, max_v)
    # heatmap_datasets(second_subplot, 2, plt.subplot(1, 2, 2), min_v, max_v)
    # # plt.tight_layout()
    plt.show()


def plot_auc_for_batches(df):
    datasets = {
        "Tweet_emotion_detection": "Twitter\nSentiment",
        "Gibberish": "Gibberish\n",
        "Amazon Review 3, 5": "Amazon\nReviews 3, 5",
        "Amazon Review 1, 5": "Amazon\nReviews 1, 5",
        "Fake news detection": "Fake News\nDetection",
    }
    algorithms = {
        "nn_min_margin": "HAC\nMin-margin",
        "mc_dropout_hac_entropy": "MC Dropout\nHAC Entropy",
        "mc_dropout_hac_bald": "MC Dropout\nHAC BALD",
        "mc_dropout_entropy": "MC Dropout\nEntropy",
        "mc_dropout_bald": "MC Dropout\nBALD",
        "mc_dropout_random": "MC Dropout\nRandom",
        # "denfi_hac_entropy": "DEnFi\nHAC Entropy",
        # "denfi_hac_bald": "DEnFi\nHAC BALD",
        # "denfi_entropy": "DEnFi\nEntropy",
        # "denfi_bald": "DEnFi\nBALD",
        # "denfi_random": "DEnFi\nRandom",
        "nn_warm_start_hac_entropy": "NN HAC Entropy\nWarm-start",
        "nn_warm_start_entropy": "NN Entropy\nWarm-start",
        "nn_warm_start_random": "NN Random\nWarm-start",
    }
    batches = [10, 20, 50, 100]
    for i, (dataset, dataset_title) in enumerate(datasets.items()):
        ax = plt.subplot(1, 5, i + 1)
        for algo, algo_title in algorithms.items():
            auc_means = [
                round(
                    df[(df["dataset"] == dataset) & (df["algorithm"] == algo) & (df["batch"] == batch)][
                        "results_mean"
                    ].tolist()[0][-1],
                    3,
                )
                for batch in batches
            ]
            ax.plot([1, 2, 3, 4], auc_means, linestyle="-", marker="+", markevery=1, lw=1, label=algo_title)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Batch size")
        ax.set_xlim(0.5, 4.5)
        if i == 0:
            ax.set_ylabel("AUC", fontsize=13.5)
        ax.set_title(dataset_title, fontsize=15)
        ax.set_xticklabels([" ", 10, 20, 50, 100], fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc="lower center", ncol=5, fontsize=11)
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def main():
    collection_results = get_collection_results()
    df = merge_resuls_and_metadata(collection_results, DATA)
    df = df[(df["algorithm"] != "nn_warm_start_hac_bald") & (df["algorithm"] != "nn_warm_start_bald")]

    plot_auc_for_batches(df)

    # Without DEnFi
    df_r = df[df["algorithm"].str.match(r"^denfi") != True]  # In case u want to exclude DEnFi

    batch_ranks = get_batch_rank(df_r)
    # plot_batch_evolutions(batch_ranks)
    # heatmap_batch(batch_ranks)

    get_batch_rank_subplots(df_r)

    # batch_ranks[0].to_clipboard(header=False, index=False)

    # dataset_ranks = get_dataset_rank(df_r)
    # heatmap_datasets(dataset_ranks)

    # With data that include DEnFi
    df_denfi = df[
        (df["batch"] == 20) & (df["dataset"] != "Gibberish") & (df["dataset"] != "Amazon Review 1, 5")
    ]
    dataset_ranks = get_dataset_rank(df_denfi)
    heatmap_datasets(dataset_ranks)
    # dataset_ranks[0].to_clipboard(header=False, index=False)


if __name__ == "__main__":
    main()
