import numpy as np

from PeonyPackage.PeonyDb import MongoDb
from typing import List, Dict, Any, Union, overload
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


def transform_text(sample: Dict[str, Any]) -> str:

    text = " ".join(
        [sample["record"]["text"]["title"], sample["record"]["text"]["body"]]
    )
    # tokens = lemmatizer(stop_words_filter(tokenizer(text)))
    # tokens = stop_words_filter(tokenizer(text))
    # features = get_word_embeddings(tokens)
    return text


def transform_label(sample: str) -> int:
    return LABEL_ENCODING[sample]


def transform(data: Any) -> np.ndarray:
    if isinstance(data[0], dict):
        transformed_data = [transform_text(sample) for sample in tqdm(data)]
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer(smooth_idf=False)
        counts = vectorizer.fit_transform(transformed_data)
        return transformer.fit_transform(counts)
    else:
        transformed_data = [transform_label(sample) for sample in tqdm(data)]
        return np.matrix(transformed_data).transpose()
