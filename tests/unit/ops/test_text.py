import unittest

from src.ops.text import center_text_around_keywords


class TestOpsText(unittest.TestCase):

    def test_center_text_around_keywords(self):

        text = ""
        expected = ""
        result = center_text_around_keywords(text, keywords=['A'])
        self.assertEqual(result, expected)

        text = "This is A sentence"
        expected = "is A sentence"
        result = center_text_around_keywords(text, keywords=['A'], radius=1)
        self.assertEqual(result, expected)

        text = "This is A's sentence"
        expected = "is A's"
        result = center_text_around_keywords(text, keywords=['A'], radius=1)
        self.assertEqual(result, expected)

        text = "The secretary of state must make sure the report is published"
        expected = "secretary of state must make sure the"
        result = center_text_around_keywords(text, keywords=['must'], radius=3)
        self.assertEqual(result, expected)

        text = "The secretary of state must make sure the report is published and it also must supervise"
        expected = "secretary of state must make sure the"
        result = center_text_around_keywords(text, keywords=['must'], radius=3)
        self.assertEqual(result, expected)

        text = "The secretary of state may make sure the report is published and it also must supervise"
        expected = "secretary of state may make sure the"
        result = center_text_around_keywords(text, keywords=['must', 'may'], radius=3)
        self.assertEqual(result, expected)

        text = "The secretary of state may make sure the report is published and it also must supervise"
        expected = "secretary of state may make sure the"
        result = center_text_around_keywords(text, keywords=['must', 'may'], radius=3)
        self.assertEqual(result, expected)

        text = "The secretary of state may make sure the report is published and it also must supervise"
        expected = "secretary of state may make"
        result = center_text_around_keywords(text, keywords=['must', 'may'], n_before=3, n_after=1)
        self.assertEqual(result, expected)

        text = "The secretary of state may make sure the report is published and it also must supervise"
        expected = "state may make sure the"
        result = center_text_around_keywords(text, keywords=['must', 'may'], n_before=1, n_after=3)
        self.assertEqual(result, expected)

    def test_center_text_around_keywords_return_spans(self):

        text = ""
        expected = "", (0, 0)
        result = center_text_around_keywords(text, keywords=['A'], spans=True)
        self.assertEqual(result, expected)

        text = "This is A sentence"
        expected = "is A sentence", (1, 4)
        result = center_text_around_keywords(text, keywords=['A'], radius=1, spans=True)
        self.assertEqual(result, expected)

        text = "This is A's sentence"
        expected = "is A's", (1, 4)
        result = center_text_around_keywords(text, keywords=['A'], radius=1, spans=True)
        self.assertEqual(result, expected)

        text = "The secretary of state must make sure the report is published"
        expected = "secretary of state must make sure the", (1, 8)
        result = center_text_around_keywords(text, keywords=['must'], radius=3, spans=True)
        self.assertEqual(result, expected)

        text = "The secretary of state may make sure the report is published and it also must supervise"
        expected = "secretary of state may make", (1, 6)
        result = center_text_around_keywords(text, keywords=['must', 'may'], n_before=3, n_after=1, spans=True)
        self.assertEqual(result, expected)

        text = "The secretary of state may make sure the report is published and it also must supervise"
        expected = "state may make sure the", (3, 8)
        result = center_text_around_keywords(text, keywords=['must', 'may'], n_before=1, n_after=3, spans=True)
        self.assertEqual(result, expected)
