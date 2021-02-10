import numpy as np
from typing import List
from typing import Tuple

from functools import reduce


def take_single_entity_for_each_class(predictions: List[Tuple[str, int, np.array]], base_class: int = 0) -> List[Tuple[str, int]]:
    """
    Restricts the NER system output to one text span per class.
    If the output contains more than one entity for a given class, we take the one
    whose average confidence (probability) score is higher

    An entity is defined by any contiguous span of tokens [t0, t1 ..., tn]
    Returns a list of tokens with their associated processed classes
    """

    def reducer(acc, x):
        i, xs = x
        token, pred, probs = xs

        if pred not in acc:
            acc[pred] = [[i]]
        else:
            last = acc[pred][-1]
            if last[-1] == i - 1:
                acc[pred][-1].append(i)
            else:
                acc[pred].append([i])

        return acc

    groups = reduce(reducer, enumerate(predictions), dict())

    groups = ((k, v, [[predictions[i][-1] for i in xs] for xs in v]) for k, v in groups.items())
    groups = ((k, v, [np.array(p) for p in ps]) for k, v, ps in groups)
    groups = ((k, v, [a.mean(axis=0) for a in ax]) for k, v, ax in groups)
    groups = ((k, v, [a[k] for a in ax]) for k, v, ax in groups)
    groups = ((k, v, np.argmax(ax)) for k, v, ax in groups)

    keep = dict(((k, set(v[a])) for k, v, a in groups))

    result = []
    for i, (token, pred, _) in enumerate(predictions):
        if (pred == base_class) or (i in keep[pred]):
            new_pred = pred
        else:
            new_pred = base_class
        result.append((token, new_pred))

    return result