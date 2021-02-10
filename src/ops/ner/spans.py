from typing import List
from typing import Tuple
from typing import Dict

from functools import reduce
from itertools import repeat


def predictions_to_spans(predictions: List[Tuple[str, int]], labels: Dict, base_class: int = 0) -> List[Dict[str, int]]:
    """
    Convert the predictions of the NER system to a list of relevant spans
    compatible with Prodigy's pre annotated data API

    For a NER-type task prodigy expects pre-annotated spans with the following schema:
    {
        start, end,
        token_start, token_end,
        label
    }

    We consider spans distinct from the base class
    """
    idx_to_label = {v: k for k, v in labels.items()}

    def reducer(acc, x):
        i, xs = x
        token, pred = xs

        if pred not in acc:
            acc[pred] = [[i]]
        else:
            last = acc[pred][-1]
            if last[-1] == i - 1:
                acc[pred][-1].append(i)
            else:
                acc[pred].append([i])

        return acc

    tokens = [a for a, _ in predictions]

    groups = reduce(reducer, enumerate(predictions), dict())
    groups = ((k, v) for k, v in groups.items() if k != base_class)
    groups = (list(zip(repeat(k), v)) for k, v in groups)
    groups = sum(groups, [])

    spans = []
    for k, xs in groups:
        start = sum([len(i) for i in tokens[:xs[0]]]) + len(tokens[:xs[0]])
        end = start + sum(map(len, tokens[xs[0]: xs[-1] + 1])) + len(tokens[xs[0]: xs[-1]]) - 1
        span = {
            'start': start,
            'end': end,
            'token_start': xs[0],
            'token_end': xs[-1],
            'label': idx_to_label[k]
        }
        spans.append(span)

    return spans

