from sklearn.base import BaseEstimator, TransformerMixin


class DataSubsetSelector(BaseEstimator, TransformerMixin):
    """For a given dataset, returns a subset of data for a provided key."""

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data: dict):
        return data[self.key]
