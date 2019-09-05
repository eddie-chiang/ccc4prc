import logging

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
        model = None

        # Split data into training and test sets.
        y = data_frame['code_comprehension_related']
        X = data_frame.drop(columns='code_comprehension_related')
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,  # 20%
                                                            random_state=2019,  # An arbitrary seed so the results can be reproduced
                                                            stratify=y)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # Declare data preprocessing steps.
        # pipeline = make_pipeline(preprocessing.StandardScaler(), )

        return model
