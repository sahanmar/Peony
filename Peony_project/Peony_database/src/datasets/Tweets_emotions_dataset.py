import pandas as pd
import logging

from pathlib import Path
from typing import Dict, List
from common import MongoDb, create_hash
from tqdm import tqdm


COLLECTION_NAME = "Tweets_emotions_dataset"


def transorm_data(record: Dict[str, any]) -> Dict[str, any]:
    transormed_record: dict = {}
    transormed_record["datasetName"] = COLLECTION_NAME
    transormed_record["datasetId"] = 3
    transormed_record["record"] = {}
    transormed_record["record"]["id"] = create_hash(
        [record["text"], record["date"], record["user"]]
    )
    transormed_record["record"]["text"] = {"body": record["text"]}
    transormed_record["record"]["label"] = record["target"]
    transormed_record["record"]["metadata"] = {
        "user": record["user"],
        "date": record["date"],
    }
    return transormed_record


def load_data(path: Path) -> List[dict]:
    data: list = []
    df = pd.read_csv(path, index_col=None, encoding="utf8")
    for _, row in tqdm(df.iterrows()):
        try:
            record = {
                "target": row["target"],
                "text": row["text"],
                "date": row["date"],
                "user": row["user"],
            }
            data.append(record)
        except KeyError:
            logging.warning(
                "Some fields are missing. This record was removed from dataset"
            )
    return data
