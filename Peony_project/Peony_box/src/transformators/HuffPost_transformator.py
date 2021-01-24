import numpy as np
import requests
import torch

from PeonyPackage.PeonyDb import MongoDb
from Peony_box.src.transformators.generalized_transformator import Transformator
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from Peony_box.src.transformators.common import (
    create_hash,
    lemmatizer,
    stop_words_filter,
    tokenizer,
)
from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from functools import lru_cache

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from bert_serving.client import BertClient

COLEECTION_NAME = "Fasttext_pretrained_embeddings"


class FastTextWordEmbeddings(Transformator):
    def __init__(self):
        super().__init__(embedding_dim=300)
        self.transformer = {}
        self.fitted: bool = False
        self.dict_length: int = 0
        self.api = MongoDb()

    def fit(self, instances: List[Dict[str, Any]], labels: List[str]) -> None:
        if self.fitted is False:
            print("transforming data...")
            transformed_data = [_transform_text(sample) for sample in tqdm(instances)]
            tokenized_text = [
                token
                for text in transformed_data
                for token in stop_words_filter(tokenizer(text))
            ]
            distinct_tokens = set(tokenized_text)
            print("creating (words -> embeddings) hash map...")
            for token in tqdm(distinct_tokens):
                embedding = self.get_embedding_from_database(token)
                if embedding is not None:
                    self.transformer[token] = embedding
            print("creating labels encoding hash map...")
            self.encoding_mapper = {
                value: index for index, value in enumerate(set(labels))
            }
            self.reverse_mapper = {
                index: value for index, value in enumerate(set(labels))
            }
            self.fitted = True
            self.dict_length = len(self.transformer.keys())

    def get_embedding_from_database(self, token: str) -> torch.Tensor:
        embedding = self.api.get_record(
            collection_name="Fasttext_pretrained_embeddings",
            collection_id=11,
            hash=create_hash([token]),
        )[0]
        if embedding is None:
            return torch.tensor([0.0 for i in range(300)])
        else:
            return torch.tensor(embedding["record"]["value"])

    def transform_instances(
        self, data: List[Dict[str, Any]]
    ) -> List[List[torch.Tensor]]:
        transformed_data = [_transform_text(sample) for sample in tqdm(data)]

        with torch.no_grad():
            transformed_instances = [
                [
                    _sentence_embed(
                        [
                            self.transformer[token]
                            for token in stop_words_filter(tokenizer(sentence))
                            if token in self.transformer
                        ]
                        + [torch.zeros((300))]
                    )
                    for sentence in sent_tokenize(text)
                ]
                for text in transformed_data
            ]
        return transformed_instances

    def transform_labels(self, data: List[str]) -> List[int]:
        return [self.transform_label(sample) for sample in tqdm(data)]

    def reset(self) -> None:
        self.transformer = {}
        self.fitted = False
        self.dict_length = 0


class LaserWordEmbeddings(Transformator):

    """
    Before you start use this transformer visit this page
    https://github.com/facebookresearch/LASER/tree/master/docker
    and run laser in docker. Use port 59012

    docker run -p 59012:80 -it laser python app.py
    """

    def __init__(self):
        super().__init__(embedding_dim=1024)

        self.url = "http://127.0.0.1:59012/vectorize"
        self.fitted: bool = False

    def transform(self, text: str) -> List[torch.Tensor]:
        return requests.get(url=self.url, params={"q": text, "lang": "en"}).json()[
            "embedding"
        ]

    def fit(self, labels: List[str]) -> None:
        if self.fitted is False:
            print("laser encoder is encoding on-prem...")
            print("creating labels encoding hash map...")
            self.encoding_mapper = {
                value: index for index, value in enumerate(set(labels))
            }
            self.reverse_mapper = {
                index: value for index, value in enumerate(set(labels))
            }
            self.fitted = True

    def transform_instances(
        self, data: List[Dict[str, Any]]
    ) -> List[List[torch.Tensor]]:
        transformed_data = [_transform_text(sample) for sample in data]

        with torch.no_grad():
            transformed_instances = [
                [torch.tensor(tensor) for tensor in self.transform(text)]
                for text in tqdm(transformed_data)
            ]
        return transformed_instances

    def transform_labels(self, data: List[str]) -> List[int]:
        return [self.transform_label(sample) for sample in tqdm(data)]

    def reset(self) -> None:
        self.fitted = False


"""
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip && unzip uncased_L-12_H-768_A-12.zip
Once we have all the files extracted in a folder, it’s time to start the BERT service:
bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len 50
"""


class BertWordEmbeddings(Transformator):
    def __init__(self):
        super().__init__(embedding_dim=768)

        self.bc_client = BertClient(ip="localhost")
        self.fitted: bool = False

    def transform(self, text: str) -> List[torch.Tensor]:
        return self.bc_client.encode(sent_tokenize(text))

    def fit(self, labels: List[str]) -> None:
        if self.fitted is False:
            print("BERT encoder is encoding on-prem...")
            print("creating labels encoding hash map...")
            self.encoding_mapper = {
                value: index for index, value in enumerate(set(labels))
            }
            self.reverse_mapper = {
                index: value for index, value in enumerate(set(labels))
            }
            self.fitted = True

    def transform_instances(
        self, data: List[Dict[str, Any]]
    ) -> List[List[torch.Tensor]]:
        transformed_data = [_transform_text(sample) for sample in data]
        with torch.no_grad():
            transformed_instances = [
                [torch.from_numpy(tensor) for tensor in self.transform(text)]
                for text in tqdm(transformed_data)
            ]
        return transformed_instances

    def transform_labels(self, data: List[str]) -> List[int]:
        return [self.transform_label(sample) for sample in tqdm(data)]

    def reset(self) -> None:
        self.fitted = False


""" 
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xzvf roberta.base.tar.gz
"""


class RoBERTaWordEmbeddings(Transformator):
    def __init__(self):
        super().__init__(embedding_dim=768)

        self.roberta = RobertaModel.from_pretrained(
            "/Users/mark/Documents/Datasets/Pretrained_models/RoBERTa/roberta.base",
            checkpoint_file="model.pt",
        )
        self.fitted: bool = False

    def transform(self, text: str) -> List[torch.Tensor]:
        return [
            _sentence_embed(
                self.roberta.extract_features(self.roberta.encode(sentence)).squeeze(0),
            )
            for sentence in self.split_sentences(text)
        ]

    def fit(self, labels: List[str]) -> None:
        if self.fitted is False:
            print("RoBERTa encoder is encoding on-prem...")
            print("creating labels encoding hash map...")
            self.encoding_mapper = {
                value: index for index, value in enumerate(set(labels))
            }
            self.reverse_mapper = {
                index: value for index, value in enumerate(set(labels))
            }
            self.fitted = True

    def transform_instances(
        self, text_documents: List[Dict[str, Any]]
    ) -> List[List[torch.Tensor]]:
        transformed_data = [
            _transform_text(text_document) for text_document in text_documents
        ]
        with torch.no_grad():
            transformed_instances = [
                self.transform(text) for text in tqdm(transformed_data)
            ]
        return transformed_instances

    def transform_labels(self, data: List[str]) -> List[int]:
        return [self.transform_label(sample) for sample in tqdm(data)]

    def reset(self) -> None:
        self.fitted = False

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        splitted_sentences: List[str] = []
        for sentence in sent_tokenize(text):
            if len(sentence) > 512:
                n_splits = len(sentence) // 512
                # not the best solution but should work...
                splitted_sentences += [
                    sentence[i * 512 : (i + 1) * 512] for i in range(n_splits)
                ]
                splitted_sentences.append(sentence[512 * n_splits :])
            else:
                splitted_sentences.append(sentence)
        return splitted_sentences


#################################
### Additional public methods ###
#################################


def _mean_agg(embeddings: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    if isinstance(embeddings, list):
        return torch.mean(torch.stack(embeddings, dim=0), dim=0)
    return torch.mean(embeddings, dim=0)


def _sentence_embed(
    embeddings: Union[torch.Tensor, List[torch.Tensor]],
    aggregator: Callable[
        [Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor
    ] = _mean_agg,
) -> torch.Tensor:
    return aggregator(embeddings)


def _normalize(embedding: Union[np.ndarray, List[float]], dim: int) -> np.ndarray:
    epsilon = 0.1
    norm = np.linalg.norm(np.asarray(embedding))
    if norm <= epsilon:
        return np.asarray([0.0 for i in range(dim)])
    else:
        return np.asarray(embedding) / norm


def _transform_text(sample: Dict[str, Any]) -> str:

    # text = " ".join(
    #     [sample["record"]["text"]["title"], sample["record"]["text"]["body"]]
    # )
    # return text
    return sample["record"]["text"]["body"]
