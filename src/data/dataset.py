import logging
import json

import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import PreTrainedTokenizer

from src.data.ner_dataset import NERDataset

from typing import Tuple
from typing import List
from typing import Dict

from src.ops.ner.preprocess import make_labels_mapping
from src.ops.sample import stratified_split

logger = logging.getLogger(__name__)


def create_datasets(tokenizer: PreTrainedTokenizer, filepath: str, split_ratios: List[float], labels: Dict, stratify_by: str = None) -> Tuple[NERDataset, NERDataset, NERDataset]:
    """
    Creates a dataset from the file `filepath`
    Each record contains two fields: 'data', 'label'
    split_ratio :: [train, val, test]
    """
    logger.info(f"Creating a dataset from file in '{filepath}'")
    df = pd.read_pickle(filepath)

    train, val, test = stratified_split(df=df, stratify_by=stratify_by, split_ratios=split_ratios)

    dataset_train = NERDataset(train, labels, tokenizer)
    dataset_test = NERDataset(test, labels, tokenizer)
    dataset_val = NERDataset(val, labels, tokenizer)

    logger.info(f"Training Set    '{len(dataset_train)}' examples")
    logger.info(f"Validation Set  '{len(dataset_val)}' examples")
    logger.info(f"Test Set        '{len(dataset_test)}' examples")

    return dataset_train, dataset_val, dataset_test

