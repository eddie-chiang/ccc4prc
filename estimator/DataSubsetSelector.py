from sklearn.base import BaseEstimator, TransformerMixin


class DataSubsetSelector(BaseEstimator, TransformerMixin):
    """For a given dataset, returns a subset of data for provided key(s)."""

    def __init__(self, keys: list):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data: dict):
        return data[self.keys]
