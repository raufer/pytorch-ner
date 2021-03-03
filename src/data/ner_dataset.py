import torch
import pandas as pd

from torch import Tensor
from pandas import DataFrame

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.constants import MAX_SEQ_LEN

from typing import Dict
from typing import List

from src.ops.ner.preprocess import create_character_mapping_from_ids_to_labels


class NERDataset(Dataset):

    def __init__(self, df: DataFrame, labels_mapping: Dict[str, int], tokenizer: PreTrainedTokenizer):
        """
        The dataframe must contain entries for:
        * text :: str
        * spans :: [dict]
          each span entry should contain:
          - label :: str
          - start :: int
          - end :: int

        These represent the locations of the span annotations for the text, along
        with its label
        """
        self.df = df
        self.labels_mapping = labels_mapping
        self.tokenizer = tokenizer

        self.encodings = tokenizer(
            list(df['text']),
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True
        )
        self.labels = [
            _make_target(text, offsets, spans, self.labels_mapping)
            for text, spans, offsets in zip(df['text'], df['spans'], self.encodings['offset_mapping'])
        ]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['target'] = torch.tensor(self.labels[idx])
        item.pop('offset_mapping', None)
        return item

    def __len__(self):
        return self.df.shape[0]


def _make_target(text: str, offset: Tensor, spans: List[Dict], labels: Dict) -> Tensor:
    """
    Creates the target for each token
    Since the tokenization process might create multiple subwords
    for a given word, we must find our target for each subword,
    i.e. for the encoded input

    To map this back to the original sentence (where the original spans
    are defined) we use the offset mappings returned by the tokenization
    method
    """
    for span in spans:
        span['label'] = labels[span['label']]

    character_mapping = create_character_mapping_from_ids_to_labels(text, spans)

    target = []

    for tpl in offset:
        a = tpl[0].item()
        b = tpl[1].item()

        if b > 0:
            b = b - 1

        if (a, b) == (0, 0):
            t = -1
        else:
            t = next((i for i in [character_mapping[a], character_mapping[b]] if i != 0), 0)

        target.append(t)

    return target

