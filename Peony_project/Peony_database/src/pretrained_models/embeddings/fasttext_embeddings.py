import io
import numpy

from pathlib import Path
from typing import Dict, List, Any
from Peony_database.src.datasets.common import create_hash
from tqdm import tqdm

COLLECTION_NAME = "Fasttext_pretrained_embeddings"


def load_data(path: Path) -> List[Dict[str, List[float]]]:
    fin = io.open(path, "r", encoding="utf-8", newline="\n", errors="ignore")
    data = []
    for line in tqdm(fin):
        tokens = line.rstrip().split(" ")
        data.append({"key": tokens[0], "value": [float(val) for val in tokens[1:]]})
    return data


def transorm_data(record: Dict[str, Any]) -> Dict[str, Any]:
    transormed_record: dict = {}
    transormed_record["datasetName"] = COLLECTION_NAME
    transormed_record["datasetId"] = 11
    transormed_record["record"] = {}
    transormed_record["record"]["id"] = create_hash([record["key"]])
    transormed_record["record"]["key"] = record["key"]
    transormed_record["record"]["value"] = record["value"]

    return transormed_record
