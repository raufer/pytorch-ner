from collections import Counter

from typing import Iterator
from typing import List


def calculate_multiclass_weights(labels: Iterator) -> List[float]:
    """
    Calculates best weights for a classification problem with various classes
    f :: Max(Number of occurrences in most common class) / (Number of occurrences in rare classes)
    """
    counter = Counter(list(labels))

    max_most_common = counter.most_common()[0][1]

    weights = [
        (value, max_most_common/count)
        for value, count in counter.most_common()
    ]
    weights = sorted(weights, key=lambda x: x[0])
    weights = [b for _, b in weights]
    return weights


def calculate_multiclass_weights_for_token_classification(targets: Iterator) -> List[float]:
    """
    Calculates best weights for a classification problem with various classes
    f :: Max(Number of occurrences in most common class) / (Number of occurrences in rare classes)

    We need to look to the distribution of the target labels to calculate
    the correct weights for each one

    This requires looking tot he already processed targets, so that we can
    count the number of occurrences of each one and extract the required
    statistics
    """
    counter = Counter(list(targets))
    most_common = [(a, b) for a, b in counter.most_common() if a != -1]

    max_most_common = most_common[0][1]

    weights = [
        (value, max_most_common/count)
        for value, count in most_common
    ]
    weights = sorted(weights, key=lambda x: x[0])
    weights = [b for _, b in weights]
    return weights

