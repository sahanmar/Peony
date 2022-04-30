import json
import pandas as pd
import argparse
import numpy as np
import re

from itertools import chain
from Peony_visualization.src.batch_active_learning_article.result_ids import DATA
from typing import Any, Dict, List


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
            .apply(lambda x: np.mean(list(chain.from_iterable(x["ranks"]))))
            .reset_index()
        )
        df_mean.columns = ["algorithm", f"{batch_size}_ranks_mean"]
        df_std = (
            df_by_datasets.groupby("algorithm", sort=False)
            .apply(lambda x: np.std(list(chain.from_iterable(x["ranks"]))))
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
            .apply(lambda x: np.mean(list(chain.from_iterable(x["ranks"]))))
            .reset_index()
        )
        df_mean.columns = ["algorithm", f"{dataset}_ranks_mean"]
        df_std = (
            df_by_datasets.groupby("algorithm", sort=False)
            .apply(lambda x: np.std(list(chain.from_iterable(x["ranks"]))))
            .reset_index()
        )
        df_std.columns = ["algorithm", f"{dataset}_ranks_std"]
        df = df_mean[["algorithm", f"{dataset}_ranks_mean"]]
        df[f"{dataset}_ranks_std"] = df_std[f"{dataset}_ranks_std"]
        return df.round(3)

    return [group_by(dataframe, dataset) for dataframe, dataset in zip(dfs, set(df["dataset"].to_list()))]


def main():
    collection_results = get_collection_results()
    df = merge_resuls_and_metadata(collection_results, DATA)
    batch_ranks = get_batch_rank(df)
    dataset_ranks = get_dataset_rank(df)
    # dataset_ranks[0].to_clipboard(header=False, index=False)

    breakpoint()


if __name__ == "__main__":
    main()
