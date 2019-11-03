import pymongo
import logging
import numpy as np

from pathlib import Path
from typing import Callable, List, Dict, Any
from tqdm import tqdm


class MongoDb:
    def __init__(
        self,
        db_user: str = "User",
        db_pass: str = "Pass",
        db_host: str = "127.0.0.1",
        db_port: int = 27017,
    ):

        url = f"mongodb://{db_user}:{db_pass}@{db_host}:{db_port}/Peony-MongoDb"
        self.client = pymongo.MongoClient(url)
        self.database = self.client["Peony-MongoDb"]

    def load_data_to_database(
        self,
        collection_name: str,
        path_to_data: Path,
        load_data: Callable[[Path], List[dict]],
        transorm_data: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        logging.info(f"extracting {collection_name}... ")
        ids: list = []
        data = load_data(path_to_data)
        logging.info("data transformation with respect to Peony database schema...")
        transormed_data = [transorm_data(record) for record in data]
        collection = self.database[collection_name]
        logging.info("uploading to Peony database...")
        for record in tqdm(transormed_data):
            ids.append(collection.insert_one(record).inserted_id)
        logging.info(
            f"{len(ids)} records from {len(data)} were successfully uploaded..."
        )

    def load_model_results(
        self,
        model: str,
        acquisition_function: str,
        algorithm_runs: int,
        learning_step: int,
        active_learning_iterations: int,
        initial_train_data_size: int,
        validation_data_size: int,
        category_1: str,
        category_2: str,
        data: List[List[float]],
        category_1_ratio: float = 0.5,
        collection_name: str = "models_results",
    ) -> None:
        results_dict: dict = {
            "model": model,
            "acquisition_function": acquisition_function,
            "algorithm_runs": algorithm_runs,
            "learning_step": learning_step,
            "active_learning_iterations": active_learning_iterations,
            "initial_train_data_size": initial_train_data_size,
            "validation_data_size": validation_data_size,
            "category_1": category_1,
            "category_2": category_2,
            "category_1_ratio": category_1_ratio,
            "category_2_ratio": 1 - (category_1_ratio),
            "results": data,
        }
        collection = self.database[collection_name]
        collection.insert_one(results_dict).inserted_id

    def get_model_results(
        self,
        filter_dict: dict,
        collection_name: str = "models_results",
        skip: int = 0,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        return list(
            self.database[collection_name].find(
                filter=filter_dict, skip=skip, limit=limit
            )
        )

    def get_record(
        self,
        collection_name: str,
        collection_id: int,
        label: str = None,
        hash: str = None,
        skip: int = 0,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        if label is not None:
            return list(
                self.database[collection_name].find(
                    filter={
                        "datasetName": collection_name,
                        "datasetId": collection_id,
                        "record.label": label,
                    },
                    skip=skip,
                    limit=limit,
                )
            )
        elif hash is not None:
            return [
                self.database[collection_name].find_one(
                    filter={
                        "datasetName": collection_name,
                        "datasetId": collection_id,
                        "record.id": hash,
                    },
                    skip=skip,
                )
            ]
        else:
            return list(
                self.database[collection_name].find(
                    filter={"datasetName": collection_name, "datasetId": collection_id},
                    skip=skip,
                    limit=limit,
                )
            )
