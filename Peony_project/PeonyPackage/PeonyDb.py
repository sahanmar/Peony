import pymongo
import logging

from pathlib import Path
from typing import Callable, List, Dict
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
        transorm_data: Callable[[Dict[str, any]], Dict[str, any]],
    ):
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

    def get_record(
        self,
        collection_name: str,
        collection_id: int,
        label: str = None,
        hash: str = None,
        skip: int = 0,
        limit: int = 10000,
    ) -> List[Dict[str, any]]:
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
            return list(
                self.database[collection_name].find_one(
                    filter={
                        "datasetName": collection_name,
                        "datasetId": collection_id,
                        "record.id": hash,
                    },
                    skip=skip,
                )
            )
        else:
            return list(
                self.database[collection_name].find(
                    filter={"datasetName": collection_name, "datasetId": collection_id},
                    skip=skip,
                    limit=limit,
                )
            )
