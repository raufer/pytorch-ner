import spacy

from typing import List
from typing import Union
from typing import Tuple


nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'textcat'])


def center_text_around_keywords(text: str, keywords: List[str], radius: int = 10, n_before: int = None, n_after: int = None, spans=False) -> Union[str, Tuple[str, Tuple]]:
    """
    Cuts the text down to be centered around certain keywords
    allowing for a certain radius before and after the keyword

    * uses as the center the first keyword that matches a token
    * we can use either use a single number for the radius or customize
    the field to consider before and after the center

    If `spans` is True also returns the start and end of the span
    """

    if (n_before is None) or (n_after is None):
        a, b = radius, radius
    else:
        a, b = n_before, n_after

    doc = nlp(text)

    keywords = set(keywords)

    center = next((token.i for token in doc if token.text in keywords), None)

    if center is None:
        return text, (0, len(doc))

    start = max(center-a, 0)
    end = min(center + b + 1, len(doc))

    centered_text = doc[start:end].text

    if spans:
        return centered_text, (start, end)
    else:
        return centered_text



