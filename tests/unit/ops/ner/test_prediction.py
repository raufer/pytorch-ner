import torch
import unittest

from src.constants import MODEL_NAME
from src.models import make_model
from src.ops.ner.prediction import subwords_class_probabilities_to_original_text, \
    subwords_class_probabilities_to_original_text_with_scores
from src.ops.ner.prediction import predict_word_classes
from src.tokenizer import make_roberta_tokenizer


class TestOpsNERPredict(unittest.TestCase):

    def test_subwords_class_probabilities_to_original_text_average(self):

        sentence = "The aerodynamics of the viature are well studied"
        encoded_tokens = ['<s>', 'The', 'Ġaer', 'odynamics', 'Ġof', 'Ġthe', 'Ġvi', 'ature', 'Ġare', 'Ġwell', 'Ġstudied', '</s>']
        input_ids = [0, 133, 16482, 41203, 9, 5, 12987, 18830, 32, 157, 8069, 2]
        offset_mapping = [
            (0, 0),
            (0, 3),
            (4, 7),
            (7, 16),
            (17, 19),
            (20, 23),
            (24, 26),
            (26, 31),
            (32, 35),
            (36, 40),
            (41, 48),
            (0, 0)
        ]

        probs = [
            [0.1772, 0.3454, 0.6675],
            [0.6001, 0.3614, 0.9699],
                [0.2356, 0.4113, 0.1761],
                [0.2445, 0.9833, 0.6258],
            [0.0850, 0.0293, 0.8442],
            [0.9242, 0.4379, 0.2544],
                [0.2707, 0.8785, 0.123],
                [0.5134, 0.3527, 0.1784],
            [0.0394, 0.7658, 0.9045],
            [0.5134, 0.3527, 0.1784],
            [0.0394, 0.7658, 0.9045],
            [0.0684, 0.9502, 0.7810]
        ]

        result = subwords_class_probabilities_to_original_text(sentence, offset_mapping, probs)

        self.assertListEqual([b for a, b in result], [2, 1, 2, 0, 1, 2, 0, 2])
        self.assertListEqual([a for a, b in result], ['The', 'aerodynamics', 'of', 'the', 'viature', 'are', 'well', 'studied'])

    def test_subwords_class_probabilities_to_original_text_average_return_probs(self):

        sentence = "The aerodynamics of the viature are well studied"
        encoded_tokens = ['<s>', 'The', 'Ġaer', 'odynamics', 'Ġof', 'Ġthe', 'Ġvi', 'ature', 'Ġare', 'Ġwell', 'Ġstudied', '</s>']
        input_ids = [0, 133, 16482, 41203, 9, 5, 12987, 18830, 32, 157, 8069, 2]
        offset_mapping = [
            (0, 0),
            (0, 3),
            (4, 7),
            (7, 16),
            (17, 19),
            (20, 23),
            (24, 26),
            (26, 31),
            (32, 35),
            (36, 40),
            (41, 48),
            (0, 0)
        ]

        probs = [
            [0.1772, 0.3454, 0.6675],
            [0.6001, 0.3614, 0.9699],
            [0.2356, 0.4113, 0.1761],
            [0.2445, 0.9833, 0.6258],
            [0.0850, 0.0293, 0.8442],
            [0.9242, 0.4379, 0.2544],
            [0.2707, 0.8785, 0.123],
            [0.5134, 0.3527, 0.1784],
            [0.0394, 0.7658, 0.9045],
            [0.5134, 0.3527, 0.1784],
            [0.0394, 0.7658, 0.9045],
            [0.0684, 0.9502, 0.7810]
        ]

        result = subwords_class_probabilities_to_original_text_with_scores(sentence, offset_mapping, probs)

        self.assertListEqual([b for a, b, c in result], [2, 1, 2, 0, 1, 2, 0, 2])
        self.assertListEqual([a for a, b, c in result], ['The', 'aerodynamics', 'of', 'the', 'viature', 'are', 'well', 'studied'])

        for a, b, c in result:
            self.assertTrue(len(c) == 3)
            for i in c:
                self.assertTrue(i <= 1)

    def test_predict_word_classes(self):

        tokenizer = make_roberta_tokenizer()

        model = make_model(MODEL_NAME.ROBERTA)(
            n_outputs=3
        )

        texts = [
            "The aerodynamics of the viature are well studied",
            "The court officer was driving away"
        ]

        predictions = predict_word_classes(model, tokenizer, texts)

        self.assertTrue(len(predictions) == 2)
        self.assertTrue(len(predictions[0]) == 8)
        self.assertTrue(len(predictions[1]) == 6)

        self.assertListEqual([a for a, b in predictions[0]], ['The', 'aerodynamics', 'of', 'the', 'viature', 'are', 'well', 'studied'])
        self.assertListEqual([a for a, b in predictions[1]], ['The', 'court', 'officer', 'was', 'driving', 'away'])

        self.assertTrue(len([b for a, b in predictions[0] if (b > 2) or (b < 0)]) == 0)
        self.assertTrue(len([b for a, b in predictions[1] if (b > 2) or (b < 0)]) == 0)


