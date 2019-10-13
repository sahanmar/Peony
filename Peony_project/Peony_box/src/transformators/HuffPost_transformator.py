import numpy as np

from PeonyPackage.PeonyDb import MongoDb
from Peony_box.src.transformators.generalized_transformator import Transformator
from typing import List, Dict, Any, Union, Optional, Tuple
from Peony_box.src.transformators.common import (
    create_hash,
    lemmatizer,
    stop_words_filter,
    tokenizer,
)
from tqdm import tqdm
from functools import lru_cache

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

COLEECTION_NAME = "Fasttext_pretrained_embeddings"
LABEL_ENCODING = {
    "POLITICS": 1,
    "WELLNESS": 2,
    "ENTERTAINMENT": 3,
    "TRAVEL": 4,
    "STYLE & BEAUTY": 5,
    "PARENTING": 6,
    "HEALTHY LIVING": 7,
    "QUEER VOICES": 8,
    "FOOD & DRINK": 9,
    "BUSINESS": 10,
    "COMEDY": 11,
    "SPORTS": 12,
    "BLACK VOICES": 13,
    "HOME & LIVING": 14,
    "PARENTS": 15,
    "THE WORLDPOST": 16,
    "WEDDINGS": 17,
    "WOMEN": 18,
    "IMPACT": 19,
    "DIVORCE": 20,
    "CRIME": 21,
    "MEDIA": 22,
    "WEIRD NEWS": 23,
    "GREEN": 24,
    "WORLDPOST": 25,
    "RELIGION": 26,
    "STYLE": 27,
    "SCIENCE": 28,
    "WORLD NEWS": 29,
    "TASTE": 30,
    "TECH": 31,
    "MONEY": 32,
    "ARTS": 33,
    "FIFTY": 34,
    "GOOD NEWS": 35,
    "ARTS & CULTURE": 36,
    "ENVIRONMENT": 37,
    "COLLEGE": 38,
    "LATINO VOICES": 39,
    "CULTURE & ARTS": 40,
    "EDUCATION": 41,
}


# def normalize(embedding: Union[np.ndarray, List[float]]) -> np.ndarray:
#     return np.asarray(embedding) / np.linalg.norm(np.asarray(embedding))


# @lru_cache(maxsize=1000)
# def get_embedding_from_database(api: MongoDb, token: str) -> np.ndarray:
#     return api.get_record(
#         collection_name="Fasttext_pretrained_embeddings",
#         collection_id=11,
#         hash=create_hash([token]),
#     )[0]


# def get_word_embeddings(tokens: List[str]) -> np.ndarray:
#     api = MongoDb()
#     word_embeddings = []
#     for token in tokens:
#         embedding = get_embedding_from_database(api, token)
#         if embedding is not None:
#             word_embeddings.append(embedding["record"]["value"])

#     return normalize(
#         np.sum([normalize(embedding) for embedding in word_embeddings], axis=0)
#     )


class HuffPostTransform(Transformator):
    def __init__(self):
        self.transformer = TfidfTransformer(smooth_idf=False)
        self.vectorizer = CountVectorizer()
        self.fitted = False
        self.dict_length = None

    def transform_instances(self, data: List[Dict[str, Any]]) -> np.ndarray:
        transformed_data = [self._transform_text(sample) for sample in tqdm(data)]
        if self.fitted is False:
            counts = self.vectorizer.fit_transform(transformed_data)
            self.transformer.fit_transform(counts)
            self.fitted = True
            self.dict_length = len(self.vectorizer.get_feature_names())
        counts = self.vectorizer.transform(transformed_data)
        return self.transformer.transform(counts)

    def transform_labels(self, data: List[str]) -> np.ndarray:
        transformed_data = [self._transform_label(sample) for sample in tqdm(data)]
        return np.asarray(transformed_data).ravel()

    @staticmethod
    def _transform_text(sample: Dict[str, Any]) -> str:

        text = " ".join(
            [sample["record"]["text"]["title"], sample["record"]["text"]["body"]]
        )
        return text

    @staticmethod
    def _transform_label(sample: str) -> int:
        return LABEL_ENCODING[sample]
