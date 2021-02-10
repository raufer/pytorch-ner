import torch
import unittest
import dictdiffer

from numpy import array

from src.ops.ner.spans import predictions_to_spans


class TestOpsNERSpans(unittest.TestCase):

    def test_predicitons_to_spans(self):

        labels = {
            'A': 0,
            'B': 1,
            'C': 2
        }
        self.assertListEqual(predictions_to_spans([], labels), [])

        predictions = [
            ('The', 0),
            ('relevant', 1),
            ('authorities', 1),
            ('might', 0),
            ('come', 0),
            ('to', 2),
            ('your', 2),
            ('door', 2)
        ]
        expected = [
            {
                'start': 4,
                'end': 23,
                'token_start': 1,
                'token_end': 2,
                'label': 'B'
            },
            {
                'start': 36,
                'end': 47,
                'token_start': 5,
                'token_end': 7,
                'label': 'C'
            }
        ]
        result = predictions_to_spans(predictions, labels)
        self.assertListEqual(result, expected)

        predictions = [
            ('The', 0),
            ('relevant', 1),
            ('authorities', 1),
            ('might', 0),
            ('come', 0),
            ('to', 1),
            ('your', 1),
            ('door', 1)
        ]
        expected = [
            {
                'start': 4,
                'end': 23,
                'token_start': 1,
                'token_end': 2,
                'label': 'B'
            },
            {
                'start': 36,
                'end': 47,
                'token_start': 5,
                'token_end': 7,
                'label': 'B'
            }
        ]
        result = predictions_to_spans(predictions, labels)
        self.assertListEqual(result, expected)

