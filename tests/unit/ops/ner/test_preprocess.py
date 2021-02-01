import unittest

from src.ops.ner import NO_LABEL_IX
from src.ops.ner.preprocess import create_character_mapping_from_ids_to_labels, make_labels_mapping


class TestOpsNERPreprocess(unittest.TestCase):

    def test_create_mapping_from_ids_to_labels(self):

        sentence = "the court officer must drive away"
        spans = []
        result = create_character_mapping_from_ids_to_labels(sentence, spans)
        expected = {
            **{i: 0 for i in range(0, len(sentence))}
        }
        self.assertDictEqual(result, expected)

        sentence = "the court officer must drive away"
        spans = [{
           'end': 17,
           'label': 1,
           'start': 0,
           'token_end': 1,
           'token_start': 0
        }]
        result = create_character_mapping_from_ids_to_labels(sentence, spans)
        expected = {
            **{i: 1 for i in range(0, 17)},
            **{i: 0 for i in range(17, len(sentence))}
        }
        self.assertDictEqual(result, expected)

        sentence = "the court officer must drive away if applicable"
        spans = [
            {
                'end': 17,
                'label': 1,
                'start': 0,
                'token_end': 1,
                'token_start': 0
            },
            {
                'end': 47,
                'label': 2,
                'start': 34,
                'token_end': 1,
                'token_start': 0
            }
        ]
        result = create_character_mapping_from_ids_to_labels(sentence, spans)

        expected = {
            **{i: 1 for i in range(0, 17)},
            **{i: 0 for i in range(17, 34)},
            **{i: 2 for i in range(34, len(sentence))}
        }
        self.assertDictEqual(result, expected)

    def test_make_labels_mapping(self):

        result = make_labels_mapping([])
        expected = {NO_LABEL_IX: 0}
        self.assertDictEqual(result, expected)

        spans = [{
            'end': 17,
            'label': 'Subject',
            'start': 0,
            'token_end': 1,
            'token_start': 0
        }]
        spans_list = [spans]
        result = make_labels_mapping(spans_list)
        expected = {NO_LABEL_IX: 0, 'Subject': 1}
        self.assertDictEqual(result, expected)

        spans_list = []
        spans_list.append([
            {
                'end': 17,
                'label': 'Detail',
                'start': 0,
                'token_end': 1,
                'token_start': 0
            }
        ])
        spans_list.append([
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
        ])
        result = make_labels_mapping(spans_list)
        expected = {NO_LABEL_IX: 0, 'Detail': 1, 'Subject': 2}
        self.assertDictEqual(result, expected)

        spans_list = []
        spans_list.append([
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
        ])
        spans_list.append([
            {
                'end': 17,
                'label': 'Detail',
                'start': 0,
                'token_end': 1,
                'token_start': 0
            }
        ])
        result = make_labels_mapping(spans_list)
        expected = {NO_LABEL_IX: 0, 'Subject': 1, 'Detail': 2}
        self.assertDictEqual(result, expected)
