import json
import logging

from pathlib import Path
from typing import Dict, List
from common import create_hash
from tqdm import tqdm


COLLECTION_NAME = "4_newsgroups_dataset"


def transorm_data(record: Dict[str, any]) -> Dict[str, any]:
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
                    logging.warning(
                        "Some fields are missing. This record was removed from dataset"
                    )
    return data
