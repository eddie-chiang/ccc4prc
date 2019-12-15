import nltk
import pandas

from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


class PosTagEstimator(BaseEstimator, TransformerMixin):
    """An estimator using NLTK part-of-speech tagging.
    Adapted from: https://www.kaggle.com/metadist/work-like-a-pro-with-pipelines-and-feature-unions

    Args:
        tokenizer: custom tokenizer function.
        normalise (bool): True - divide all values by the total number of tags in the sentence.
    """

    def __init__(self, tokenizer=lambda x: x.split(), normalize=True):
        self.tokenizer = tokenizer
        self.normalize = normalize

    def fit(self, X, y=None):
        """This a transformer class, so this method does not do much. """
        return self

    def transform(self, X):
        X_tagged = X.apply(self.__pos_tag).apply(pandas.Series).fillna(0)
        X_tagged['n_tokens'] = X_tagged.apply(sum, axis=1)
        if self.normalize:
            X_tagged = X_tagged.divide(X_tagged['n_tokens'], axis=0)

        return X_tagged

    def __pos_tag(self, sentence):
        """Tokenize and count parts of speech. """
        return Counter(tag for word, tag in nltk.pos_tag(self.tokenizer(sentence)))
