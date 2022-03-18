import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from Peony_database.src.datasets.common import create_hash
from tqdm import tqdm


COLLECTION_NAME = "gibberish"
COLLECTION_ID = 10

LABEL_MAP = {"__label__1": 1, "__label__2": 2}


def transorm_data(record: Dict[str, Any]) -> Dict[str, Any]:
    transormed_record: dict = {}
    transormed_record["datasetName"] = COLLECTION_NAME
    transormed_record["datasetId"] = COLLECTION_ID
    transormed_record["record"] = {}
    transormed_record["record"]["id"] = create_hash([record["title"]])
    transormed_record["record"]["text"] = {
        "title": record["title"],
        "body": record["body"],
    }
    transormed_record["record"]["label"] = record["label"]
    return transormed_record


def load_data(path: Path) -> List[dict]:
    data: list = []

    # Snippet to see problematic encoding
    # import chardet
    # with open(file, 'rb') as rawdata:
    #     result = chardet.detect(rawdata.read(100000))
    # result

    df = pd.read_csv(path, names=["label", "text"], header=None, encoding="ISO-8859-1")
    for _, row in df.iterrows():
        data.append(
            {
                "title": "",
                "body": row["text"],
                "label": LABEL_MAP[row["label"]],
            }
        )
    return data
