import json

from pathlib import Path
from typing import Dict, List, Any
from Peony_database.src.datasets.common import create_hash
from tqdm import tqdm


COLLECTION_NAME = "HuffPost_dataset"
COLLECTION_ID = 1


def transorm_data(record: Dict[str, Any]) -> Dict[str, Any]:
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
