import json

from pathlib import Path
from typing import Dict, List
from common import MongoDb, create_hash
from tqdm import tqdm


COLLECTION_NAME = "200k_news_dataset"


def transorm_data(record: dict) -> dict:
    transormed_record: dict = {}
    transormed_record["datasetName"] = COLLECTION_NAME
    transormed_record["datasetId"] = 1
    transormed_record["record"] = {}
    transormed_record["record"]["id"] = create_hash(
        [record["title"], record["metadata"]["short_description"]]
    )
    transormed_record["record"]["snippet"] = record["metadata"]["short_description"]
    transormed_record["record"]["text"] = {
        "title": record["title"],
        "body": record["body"],
    }
    transormed_record["record"]["label"] = record["metadata"]["category"]
    transormed_record["record"]["metadata"] = {
        "authors": record["metadata"]["authors"],
        "language": record["language"],
    }
    return transormed_record


def load_data(path: Path) -> List[dict]:
    data: list = []
    for json_doc in tqdm(path.iterdir()):
        if json_doc.stem != ".DS_Store":
            with open(json_doc, "r") as f:
                lines = f.readlines()
            data.extend(*[json.loads(line) for line in lines])
    return data


def load_data_to_database(path_to_data: Path):

    print("extracting 200k_news_dataset... ")
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
