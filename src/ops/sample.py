import pandas as pd

from pandas import DataFrame

from typing import List
from typing import Tuple

from sklearn.model_selection import train_test_split


def stratified_split(df: DataFrame, split_ratios: List[float], stratify_by: str = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Splits a dataframe into train validation and test

    split_ratio :: [train, val, test]
    """
    train_size, val_size, test_size = split_ratios
    X = df

    if stratify_by is not None:
        raise NotImplementedError("Implement 'stratify by'")

    X_train, X_test = train_test_split(X, train_size=train_size, random_state=11)
    test_size_renormalized = test_size / (test_size + val_size)
    X_test, X_val = train_test_split(X_test, train_size=test_size_renormalized, random_state=11)

    df_train = pd.concat([X_train], axis=1)
    df_val = pd.concat([X_val], axis=1)
    df_test = pd.concat([X_test], axis=1)

    return df_train, df_val, df_test


if __name__ == '__main__':

    df = pd.read_csv('/Users/raulferreira/pytorch-roberta-classifier/data/fake_news_sample.csv')

    train, val, test = stratified_split(df, 'label', split_ratios=[0.7, 0.2, 0.1])

