import torch
import spacy
import numpy as np

from typing import List
from typing import Tuple
from functools import reduce
from collections import defaultdict

from torch.utils.data import DataLoader

from src.constants import BATCH_SIZE
from src.data.simple_ner_dataset import SimpleNERDataset
from src import device

nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'textcat'])


def subwords_class_probabilities_to_original_text(text: str, offset_mapping: List[Tuple[int, int]], probs: List[List[float]]) -> List[Tuple[str, int]]:
    """
    Returns the aggregated class prediction at the original sentence word level

    A NER system the model predicts tag annotations on the sub-word level,
    not on the word level. To obtain word-level annotations,
    we need to aggregate the sub-word level predictions for each word.

    Aggregation method:
        *  average the classes of subword predictions

    Returns a list of pairs :: (token, class)
    """
    doc = nlp(text)

    char_offsets = [(token.idx, token.idx + len(token), token.i) for token in doc]

    if len(offset_mapping) != len(probs):
        raise AssertionError(f"Number of subwords offsets is different from the number of subword predicitons '{len(offset_mapping)} != {len(probs)}'")

    char_index_to_token_index_mapping = reduce(
        lambda acc, x: {**acc, **x},
        [{i: c for i in range(a, b + 1)} for a, b, c in char_offsets],
        dict()
    )

    xs = ((offset, probs) for offset, probs in zip(offset_mapping, probs) if (offset != (0, 0)))
    xs = ((char_index_to_token_index_mapping[b], probs) for (a, b), probs in xs)

    data = defaultdict(lambda : [])
    for token_idx, probs in xs:
        data[token_idx].append(probs)

    data = {k: np.array(v).mean(axis=0) for k, v in data.items()}
    data = [(k, np.argmax(v)) for k, v in data.items()]

    predictions = [b for _, b in sorted(data, key=lambda x: x[0])]
    predictions = list(zip([t.text for t in doc], predictions))
    return predictions


def subwords_class_probabilities_to_original_text_with_scores(text: str, offset_mapping: List[Tuple[int, int]], probs: List[List[float]]) -> List[Tuple]:
    """
    Returns the aggregated class prediction at the original sentence word level

    A NER system the model predicts tag annotations on the sub-word level,
    not on the word level. To obtain word-level annotations,
    we need to aggregate the sub-word level predictions for each word.

    Aggregation method:
        *  average the classes of subword predictions

    Returns a list of pairs :: (token, class)
    e.g.
    ('The', 2, array([0.6001, 0.3614, 0.9699]))
    ('aerodynamics', 1, array([0.24005, 0.6973 , 0.40095]))
    ('of', 2, array([0.085 , 0.0293, 0.8442]))
    ('the', 0, array([0.9242, 0.4379, 0.2544]))
    """
    doc = nlp(text)

    char_offsets = [(token.idx, token.idx + len(token), token.i) for token in doc]

    if len(offset_mapping) != len(probs):
        raise AssertionError(f"Number of subwords offsets is different from the number of subword predicitons '{len(offset_mapping)} != {len(probs)}'")

    char_index_to_token_index_mapping = reduce(
        lambda acc, x: {**acc, **x},
        [{i: c for i in range(a, b + 1)} for a, b, c in char_offsets],
        dict()
    )

    xs = ((offset, probs) for offset, probs in zip(offset_mapping, probs) if (offset != (0, 0)))
    xs = ((char_index_to_token_index_mapping[b], probs) for (a, b), probs in xs)

    data = defaultdict(lambda : [])
    for token_idx, probs in xs:
        data[token_idx].append(probs)

    data = {k: np.array(v).mean(axis=0) for k, v in data.items()}
    data = [(k, np.argmax(v), v) for k, v in data.items()]

    predictions = [(b, c) for _, b, c in sorted(data, key=lambda x: x[0])]
    predictions = list(zip([t.text for t in doc], predictions))
    predictions = [(a, b, c) for a, (b, c) in predictions]
    return predictions


def predict(model, iterator: DataLoader) -> Tuple[List, List]:
    """
    Given a model and a data iterator
    returns the model prediction

    y_pred :: (iter.shape[0])
    """
    y_pred = []
    y_probs = []

    model.eval()

    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop('offset_mapping', None)

            output = model(**batch)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_probs.extend(sm(output).tolist())

    return y_pred, y_probs


def predict_word_classes(model, tokenizer, texts: List[str]) -> List[List[Tuple[str, int]]]:
    """
    For each text predicts the word-level classes by aggregating and averaging
    the predictions at the subword level as dictated by the NER system

    Returns a list of integers with the word level predictions
    If `return_words` is set returns list of pairs :: (word, class)
    """
    dataset = SimpleNERDataset(texts, tokenizer)
    iterator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_pred, y_probs = predict(model, iterator)

    offset_mapping = dataset.encodings['offset_mapping']

    xs = zip(texts, offset_mapping, y_probs)

    predictions = [subwords_class_probabilities_to_original_text(t, ox.tolist(), px) for (t, ox, px) in xs]
    return predictions






