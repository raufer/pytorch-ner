from typing import List
from typing import Dict

from src.ops.ner import NO_LABEL_IX, NO_LABEL_TOKEN
from functools import reduce


def create_character_mapping_from_ids_to_labels(text: str, spans: List[Dict]) -> Dict[int, str]:
    """
    Given a sentence, and a list of relevant labelled spans,
    creates a map from character index to label

    Each entry in the span should contain:
        label :: int
        start :: int
        end :: int

    Characters that do not fall within a span are labelled with a constant index
    """
    base = {i: NO_LABEL_IX for i in range(0, len(text))}
    spans_layer = [{i: d['label'] for i in range(d['start'], d['end'])} for d in spans]
    spans_layer = reduce(lambda acc, x: {**acc, **x}, spans_layer, dict())

    mapping = {
        **base,
        **spans_layer
    }

    return mapping


def make_labels_mapping(spans: List[List[Dict]]) -> Dict[str, int]:
    """
    Constructs a mapping of labels to unique numerical indexes

    Each entry in spans is a list of spans for a given text
    text -> [spans]

    Each individual span should be a dict containing an entry
    with the key 'label'

    We traverse this nested structures in order and find all unique
    entries. Every time a new entry is found, it is added to the mapping;
    its index is constructed by adding + 1 to the highest entry
    """
    seen = set()
    ordered_labels = [NO_LABEL_TOKEN]

    for xs in spans:
        for x in xs:
            if x['label'] not in seen:
                ordered_labels.append(x['label'])
                seen.add(x['label'])
    mapping = {k: i for i, k in enumerate(ordered_labels)}
    return mapping
