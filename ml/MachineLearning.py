import logging

from pandas import DataFrame
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
        
        # Create a Naive Bayes Gaussian Classifier.
        model = GaussianNB()

        # Split data into training and test sets.
        y = data_frame['code_comprehension_related']
        X = data_frame.drop(columns='code_comprehension_related')
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,  # 20%
                                                            random_state=2019,  # An arbitrary seed so the results can be reproduced
                                                            stratify=y)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # Declare data preprocessing steps.
        # Encode features, i.e. convert string labels into numbers.
        label_encoder = preprocessing.LabelEncoder()

        dialogue_act_classification_ml = data_frame['dialogue_act_classification_ml']
        dialogue_act_classification_ml_encoded = label_encoder.fit_transform(dialogue_act_classification_ml)

        comment_is_by_author = data_frame['comment_is_by_author']
        comment_is_by_author_encoded = label_encoder.fit_transform(comment_is_by_author)

        code_comprehension_related = data_frame['code_comprehension_related']
        label = label_encoder.fit_transform(code_comprehension_related)

        # Combinig features into single listof tuples
        features = zip(dialogue_act_classification_ml_encoded, comment_is_by_author_encoded)

        # Train the model using the training sets
        model.fit(features, label)

        #Predict Output
        predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild


        # pipeline = make_pipeline(preprocessing.StandardScaler(), )

        return model
