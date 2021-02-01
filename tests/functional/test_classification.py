import unittest

import pandas as pd
from sklearn.metrics import f1_score

from src.constants import MODEL_NAME

from src.tokenizer.roberta import make_roberta_tokenizer

from src.ops.weights import calculate_multiclass_weights
from src.data.dataset import create_datasets
from src.data.iterator import make_iterators

from src.main import training_job, pipeline
from src.config import config
from src.tokenizer.xlnet import make_xlnet_tokenizer


class TestFunctionalClassification(unittest.TestCase):

    def test_ner_roberta(self):

        datapath = '/Users/raulferreira/pytorch-ner/data/prodigy-sample-annotation.pickle'
        output_dir = '/Users/raulferreira/pytorch-ner/output'

        config['num-epochs-pretrain'] = 1
        config['num-epochs-train'] = 1

        model, y_true, y_pred, output_path, train_dataset, val_dataset, test_dataset = pipeline(
            datapath=datapath,
            modelname=MODEL_NAME.ROBERTA,
            output_dir=output_dir
        )

        score = f1_score(y_true, y_pred, average='weighted')
        self.assertTrue(score <= 1.0)

