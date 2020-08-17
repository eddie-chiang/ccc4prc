from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer:
    """Adapted from: https://scikit-learn.org/stable/modules/feature_extraction.html
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
