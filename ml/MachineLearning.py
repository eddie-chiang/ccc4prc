import logging
import numpy

from pandas import DataFrame
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


class MachineLearning:
    """A machine learning class for supervised learning to create a model."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def learn(self, data_frame: DataFrame):
        """Using Scikit-learn and supervised training to create a machine learning model.

        Args:
            data_frame (DataFrame): Data with training and test sets.
        Returns:
            model: a trained machine learning model.
        """
        return self.__perform_naive_bayes(data_frame)        

    def __perform_naive_bayes(self, data_frame: DataFrame):
        # Create a Naive Bayes Gaussian Classifier.
        model = GaussianNB()

        # Declare data preprocessing steps.
        # Encode features, i.e. convert string labels into numbers.
        label_encoder = preprocessing.LabelEncoder()

        dialogue_act_classification_ml = data_frame['dialogue_act_classification_ml']
        dialogue_act_classification_ml_encoded = label_encoder.fit_transform(
            dialogue_act_classification_ml)

        comment_is_by_author = data_frame['comment_is_by_author']
        comment_is_by_author_encoded = label_encoder.fit_transform(
            comment_is_by_author)

        code_comprehension_related = data_frame['code_comprehension_related']
        label = label_encoder.fit_transform(code_comprehension_related)

        # Combinig features into single listof tuples
        features = list(zip(dialogue_act_classification_ml_encoded,
                            comment_is_by_author_encoded))

        # Split data into training and test sets.
        target = data_frame['code_comprehension_related']
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            target,
                                                            test_size=0.2,  # 20%
                                                            random_state=2019,  # An arbitrary seed so the results can be reproduced
                                                            stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # Train the model using the training sets.
        model.fit(X_train, y_train)

        # Predict the response for test set.
        y_pred = model.predict(X_test)

        # Model accuracy, how often is the classifier correct?
        self.logger.info(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
        
        for class_label in numpy.unique(y_test):
            self.logger.info(f'Precision for "{class_label}": {metrics.precision_score(y_test, y_pred, pos_label=class_label)}')
            self.logger.info(f'Recall for "{class_label}": {metrics.recall_score(y_test, y_pred, pos_label=class_label)}')        

        return model