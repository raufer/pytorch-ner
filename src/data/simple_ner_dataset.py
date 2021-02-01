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


class SimpleNERDataset(Dataset):

    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer):
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
        self.texts = texts
        self.tokenizer = tokenizer

        self.encodings = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.texts)


