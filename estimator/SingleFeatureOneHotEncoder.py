from pandas import Series
from sklearn.preprocessing import OneHotEncoder


class SingleFeatureOneHotEncoder(OneHotEncoder):
    """Subclass of OneHotEncoder to encode an 1D array with a single feature."""

    def __init__(self, categories='auto'):
        super().__init__(categories=categories)

    def fit(self, X, y=None):
        X = self.__convert_array(X)
        y = self.__convert_array(y)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        X = self.__convert_array(X)
        y = self.__convert_array(y)
        return super().fit_transform(X, y)

    def transform(self, X):
        X = self.__convert_array(X)
        return super().transform(X)

    def __convert_array(self, array):
        """Convert a 1D array to 2D array."""
        if isinstance(array, Series) and array.ndim == 1:
            array = array.values.reshape(len(array), 1)
        return array
