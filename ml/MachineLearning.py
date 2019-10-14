import logging
import numpy

from pandas import DataFrame
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


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
        # # Create a Gaussian Naive Bayes classifier.
        # model = GaussianNB()

        # # Declare data preprocessing steps.
        # # Encode features, i.e. convert string labels into numbers.
        # label_encoder = preprocessing.LabelEncoder()

        # dialogue_act_classification_ml = data_frame['dialogue_act_classification_ml']
        # dialogue_act_classification_ml_encoded = label_encoder.fit_transform(
        #     dialogue_act_classification_ml)

        # comment_is_by_author = data_frame['comment_is_by_author']
        # comment_is_by_author_encoded = label_encoder.fit_transform(
        #     comment_is_by_author)

        # # body = data_frame['body']
        # # count_vectorizer = CountVectorizer(stop_words='english')

        # # Combinig features into single listof tuples
        # features = list(zip(dialogue_act_classification_ml_encoded,
        #                     comment_is_by_author_encoded))

        # # Split data into training and test sets.
        # target = data_frame['code_comprehension_related']
        # X_train, X_test, y_train, y_test = train_test_split(features,
        #                                                     target,
        #                                                     test_size=0.2,  # 20%
        #                                                     random_state=2019,  # An arbitrary seed so the results can be reproduced
        #                                                     stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # # Train the model using the training sets.
        # model.fit(X_train, y_train)

        # # Predict the response for test set.
        # y_pred = model.predict(X_test)

        # # Model accuracy, how often is the classifier correct?
        # self.logger.info(f'{metrics.classification_report(y_test, y_pred, digits=8)}')

        # return model

        # # Create a Multinomial Naive Bayes classifier.
        # # Make a pipeline for data preprocessing.
        # # Encode features, i.e. convert string labels into numbers.
        # classifier = make_pipeline(preprocessing.OneHotEncoder(), MultinomialNB())

        # # Split data into training and test sets.
        # target = data_frame['code_comprehension_related']
        # features = data_frame[['dialogue_act_classification_ml', 'comment_is_by_author']]
        # X_train, X_test, y_train, y_test = train_test_split(features,
        #                                                     target,
        #                                                     test_size=0.2,  # 20%
        #                                                     random_state=2019,  # An arbitrary seed so the results can be reproduced
        #                                                     stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # # Train the model using the training sets.
        # classifier.fit(X_train, y_train)

        # # Predict the response for test set.
        # y_pred = classifier.predict(X_test)

        # # Model accuracy, how often is the classifier correct?
        # self.logger.info(f'{metrics.classification_report(y_test, y_pred, digits=8)}')

        # return classifier

        # Create a Multinomial Naive Bayes classifier.
        # Make a pipeline for data preprocessing.
        # Encode features, i.e. convert string labels into numbers.
        classifier = Pipeline([
            ('cv', CountVectorizer(stop_words='english')),
            # ('tfidf', TfidfTransformer()),
            ('mnb', MultinomialNB())
        ])

        # Split data into training and test sets.
        target = data_frame['code_comprehension_related']
        features = data_frame['body']
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            target,
                                                            test_size=0.2,  # 20%
                                                            random_state=2019,  # An arbitrary seed so the results can be reproduced
                                                            stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # Train the model using the training sets.
        classifier.fit(X_train, y_train)

        # Predict the response for test set.
        y_pred = classifier.predict(X_test)

        # Model accuracy, how often is the classifier correct?
        self.logger.info(
            f'{metrics.classification_report(y_test, y_pred, digits=8)}')

        # Predict the previously trained YES (code_comprehension_related).
        test = {
            '1': 'Should this commented out code still be in here?',
            '2': 'Can this be private?',
            '3': 'Can it work with "parallel: true"?',
            '4': 'I am confused, aren\'t we using `__`?',
            '5': 'No need for DatabaseJournalEntry?'
        }
        result_pred = classifier.predict(test)

        return classifier

        ## TODO Stemming
        # Example code 
        # Stemming Code
        # from nltk.stem.snowball import SnowballStemmer
        # stemmer = SnowballStemmer("english", ignore_stopwords=True)

        # class StemmedCountVectorizer(CountVectorizer):
        #     def build_analyzer(self):
        #         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        #         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
            
        # stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

        # text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
        #                             ('mnb', MultinomialNB(fit_prior=False))])

        # text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)

        # predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

        # np.mean(predicted_mnb_stemmed == twenty_test.target)

        ## TODO Lemmatization