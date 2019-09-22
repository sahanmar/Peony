import json

from pathlib import Path
from typing import Dict, List
from common import MongoDb, create_hash
from tqdm import tqdm


COLLECTION_NAME = "4_newsgroups_dataset"


def transorm_data(record: dict) -> dict:
    transormed_record: dict = {}
    transormed_record["datasetName"] = COLLECTION_NAME
    transormed_record["datasetId"] = 2
    transormed_record["record"] = {}
    transormed_record["record"]["text"] = {"body": record["text"]}
    transormed_record["record"]["label"] = record["label"]
    transormed_record["record"]["metadata"] = {"language": "en"}
    transormed_record["record"]["id"] = create_hash([record["text"]])
    return transormed_record


def load_data(path: Path) -> List[dict]:
    data: list = []
    for folder in tqdm(path.iterdir()):
        if folder.stem != ".DS_Store":
            for record in folder.iterdir():
                try:
                    with open(record, "r", encoding="utf-8") as f:
                        data.append(
                            {"text": f.read(), "label": f"{folder.stem}{folder.suffix}"}
                        )
                except:
                    pass
    return data


def load_data_to_database(path_to_data: Path):

    print("extracting 4_newsgroups_dataset... ")
    ids: list = []
    data = load_data(path_to_data)
    print("data transformation with respect to Peony database schema...")
    transormed_data = [transorm_data(record) for record in data]
    api = MongoDb()
    collection = api.databse[COLLECTION_NAME]
    print("uploading to Peony database...")
    for record in tqdm(transormed_data):
        ids.append(collection.insert_one(record).inserted_id)
    print(f"{len(ids)} records from {len(data)} were successfully uploaded...")
    print("")
