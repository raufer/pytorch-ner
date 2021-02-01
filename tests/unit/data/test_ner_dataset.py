import unittest
import pandas as pd

from src.constants import MAX_SEQ_LEN
from src.data.ner_dataset import NERDataset
from src.ops.ner.preprocess import make_labels_mapping
from src.tokenizer import make_roberta_tokenizer


class TestNERDataset(unittest.TestCase):

    def test_basic(self):

        sentences_list = []
        spans_list = []

        sentence = "the court officer must drive away"
        spans = [{
           'end': 17,
           'label': 'Subject',
           'start': 0,
           'token_end': 1,
           'token_start': 0
        }]
        sentences_list.append(sentence)
        spans_list.append(spans)

        sentence = "the court officer must drive away if applicable"
        spans = [
            {
                'end': 17,
                'label': 'Subject',
                'start': 0,
                'token_end': 1,
                'token_start': 0
            },
            {
                'end': 47,
                'label': 'Detail',
                'start': 34,
                'token_end': 1,
                'token_start': 0
            }
        ]
        sentences_list.append(sentence)
        spans_list.append(spans)

        df = pd.DataFrame(list(zip(sentences_list, spans_list)), columns=['text', 'spans'])
        labels = make_labels_mapping(spans_list)
        tokenizer = make_roberta_tokenizer()

        dataset = NERDataset(df, labels, tokenizer)

        # the court officer must drive away
        # [0, 627, 461, 1036, 531, 1305, 409, 2]
        # the court officer must drive away if applicable
        # [0, 627, 461, 1036, 531, 1305, 409, 114, 10404, 2]
        expected_target = [
            [-1, 1, 1, 1, 0, 0, 0, -1] + [-1] * (MAX_SEQ_LEN - 8),
            [-1, 1, 1, 1, 0, 0, 0, 2, 2, -1] + [-1] * (MAX_SEQ_LEN - 10)
        ]

        for t, i in zip(expected_target, dataset):
            self.assertListEqual(t, i['target'].tolist())

