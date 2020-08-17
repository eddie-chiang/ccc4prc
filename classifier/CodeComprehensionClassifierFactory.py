from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


class CodeComprehensionClassifierFactory:
    """A factory that creates a classifier to predict code comprehension related Pull Request Comments."""

    @staticmethod
    def get_classifier():
        """Defines classification algorithm, to be used in the supervised learning, to create a model.

        Returns:
            classifier: A classifier implements "fit", "score", "predict", and "predict_proba" methods.
        """

        one_hot_encoder_categories = [
            [
                'Accept',
                'Bye',
                'Clarify',
                'Continuer',
                'Emotion',
                'Emphasis',
                'Greet',
                'Other',
                'Reject',
                'Statement',
                'System',
                'whQuestion',
                'yAnswer',
                'nAnswer',
                'ynQuestion'
            ],
            [
                False,  # 0 should come before 1 for numerical columns.
                True
            ]
        ]

        column_transformer = ColumnTransformer(
            transformers=[
                ('body_bow_vectorizer', TfidfVectorizer(stop_words=None, ngram_range=(2, 2)), 'body'),
                (
                    'categorical_transformer',
                    OneHotEncoder(categories=one_hot_encoder_categories),
                    ['dialogue_act_classification_ml', 'comment_is_by_author']
                ),
            ],
            transformer_weights={
                'body_bow_vectorizer': 1,
                'categorical_transformer': 3.8,
            },
            verbose=False)

        classifier = Pipeline(
            steps=[
                ('preprocessor', column_transformer),
                ('classifier', SVC(kernel='linear', C=1.2, probability=True))],
            verbose=False)

        return classifier
