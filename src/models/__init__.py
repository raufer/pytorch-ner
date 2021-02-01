from src.constants import MODEL_NAME

from src.models.roberta_token_classifier import RobertaTokenClassifier
from src.models.xlnet_token_classifier import XLNETTokenClassifier


def make_model(modelname: str):
    """
    Returns the correct model based on the selection
    """
    if modelname == MODEL_NAME.ROBERTA:
        return RobertaTokenClassifier

    elif modelname == MODEL_NAME.XLNET:
        return XLNETTokenClassifier

    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
