from PeonyPackage.PeonyDb import MongoDb
from Peony_box.src.transformators.HuffPost_transformator import (
    HuffPostTransformWordEmbeddings,
)
from typing import Dict, Any

COLEECTION_NAME = "Tweets_emotions_dataset"


class TweetsEmotionsTransformWordEmbeddings(HuffPostTransformWordEmbeddings):
    def __init__(self):
        self.transformer = {}
        self.fitted: bool = False
        self.dict_length: int = 0
        self.api = MongoDb()
        self.encoding_mapper: Dict[int, int] = {}
        self.reverse_mapper: Dict[int, str] = {}

    @staticmethod
    def _transform_text(sample: Dict[str, Any]) -> str:
        return sample["record"]["text"]["body"]

    def transform_label(self, sample: int) -> int:
        return self.encoding_mapper[sample]

    def transform_to_label(self, value: int) -> str:
        return self.reverse_mapper[value]
