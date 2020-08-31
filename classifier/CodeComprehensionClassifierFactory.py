from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from nlp import LemmaTokenizer


class CodeComprehensionClassifierFactory:
    """A factory that creates a classifier to predict code comprehension related Pull Request Comments."""

    @staticmethod
    def get_classifier(dac_labels: list):
        """Defines classification algorithm, to be used in the supervised learning, to create a model.

        Args:
            dac_labels (list): Labels for Dialogue Act Classifications. Which should include: Accept, Bye, Clarify, Continuer, Emotion, Emphasis, Greet, Other, Reject, Statement, System, nAnswer, whQuestion, yAnswer, ynQuestion.

        Returns:
            classifier: A classifier implements "fit", "score", "predict", and "predict_proba" methods.
        """

        is_author_categories = [
            False,  # 0 should come before 1 for numerical columns.
            True
        ]

        column_transformer = ColumnTransformer(
            transformers=[
                (
                    'body_tdidf_vectorizer',
                    TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', ngram_range=(1, 2)),
                    'body'
                ),
                (
                    'dac_transformer',
                    OneHotEncoder(categories=[dac_labels]),
                    ['dialogue_act_classification_ml']
                ),
                (
                    'is_author_transformer',
                    OneHotEncoder(categories=[is_author_categories]),
                    ['comment_is_by_author']
                ),
            ],
            transformer_weights={
                'body_tdidf_vectorizer': 4,
                'dac_transformer': 1,
                'is_author_transformer': 2,
            },
            verbose=False)

        classifier = Pipeline(
            steps=[
                ('preprocessor', column_transformer),
                ('classifier', LogisticRegression(C=500000, solver='lbfgs'))
            ],
            verbose=False)

        return classifier
