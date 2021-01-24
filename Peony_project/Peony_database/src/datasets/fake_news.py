import pandas as pd

from pathlib import Path
from typing import Dict, List, Any
from Peony_database.src.datasets.common import create_hash
from tqdm import tqdm


COLLECTION_NAME = "fake_news_dataset"
COLLECTION_ID = 8


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
    transormed_record["record"]["metadata"] = {"subject": record["subject"]}
    return transormed_record


def load_data(path: Path) -> List[dict]:
    data: list = []
    for csv in tqdm(path.iterdir()):
        if csv.suffix == ".csv":
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                data.append(
                    {
                        "title": row["title"],
                        "body": row["text"],
                        "subject": row["subject"],
                        "label": csv.stem,
                    }
                )
    return data
