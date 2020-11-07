import torch

from typing import Callable, List, Dict


class Transformator:
    def __init__(
        self,
        embedding_dim: int,
        text_collate: Callable[[List[torch.Tensor]], torch.Tensor],
    ):
        self.embedding_dim = embedding_dim
        self.text_collate = text_collate

        self.encoding_mapper: Dict[str, int] = {}
        self.reverse_mapper: Dict[int, str] = {}

    def fit(self):
        pass

    def transform_instances(self):
        pass

    def transform_labels(self):
        transformed_data = [self.transform_label(sample) for sample in tqdm(data)]
        return np.asarray(transformed_data).ravel()

    def reset(self):
        pass

    def transform_label(self, sample: str) -> int:
        return self.encoding_mapper[sample]

    def transform_to_label(self, value: int) -> str:
        return self.reverse_mapper[value]
