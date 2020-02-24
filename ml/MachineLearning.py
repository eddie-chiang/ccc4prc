import logging
import nltk
import numpy

from estimator import DataSubsetSelector, PosTagEstimator, SingleFeatureOneHotEncoder
from pandas import DataFrame
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


class MachineLearning:
    """A machine learning class for supervised learning to create a model."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def active_learn(self, seed: DataFrame, test_dataset: DataFrame):
        """Using Scikit-learn, supervised training and active learning to create a machine learning model.

        Args:
            seed (DataFrame): Data with the labeled training set.
            test_dataset (DataFrame): Separate test dataset to measure the performance of the machine learning model.
        Returns:
            model: a trained machine learning model.
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
                ('body_bow_pipeline', CountVectorizer(
                    stop_words='english'), 'body'),
                #('body_ngram_pipeline', CountVectorizer(stop_words='english', ngram_range=(1, 3)), 'body'),
                ('categorical_transformer', OneHotEncoder(categories=one_hot_encoder_categories), [
                 'dialogue_act_classification_ml', 'comment_is_by_author']),
                # ('comment_is_by_author_pipeline',
                #  SingleFeatureOneHotEncoder(), 'comment_is_by_author'),
            ],
            transformer_weights={
                'body_bow_pipeline': 1.0,
                # 'body_ngram_pipeline': 0.5,
                'categorical_transformer': 1.0,
                # 'comment_is_by_author_pipeline': 0.1,
            },
            verbose=True)

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", column_transformer),
                ('classifier', SVC(kernel='linear', probability=True))],
            verbose=True)

        # Use Grid Search to perform hyper parameter tuning in order to determine the optimal values for the machine learning model.
        # TODO tweak the search params.
        # {'feature_union__tfdif_features__tfidf__use_idf': (True, False)}
        grid_search_cv_params = {}
        classifier = GridSearchCV(
            full_pipeline, grid_search_cv_params, cv=5, verbose=2)

        # Split data into training and test sets.
        target = seed['code_comprehension_related']
        features = seed[[
            'body', 'dialogue_act_classification_ml', 'comment_is_by_author']]
        X_train, sample_pool, y_train, _ = train_test_split(features,
                                                  target,
                                                  test_size=0.2,  # 20%
                                                  random_state=2019,  # An arbitrary seed so the results can be reproduced
                                                  stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        X_test = test_dataset[[
            'body', 'dialogue_act_classification_ml', 'comment_is_by_author']]
        y_test = test_dataset['code_comprehension_related']

        # Train the model using the training sets.
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        
        # Model accuracy, how often is the classifier correct?
        self.logger.info(
            f'{metrics.classification_report(y_test, y_pred, digits=8)}')

        # Pool-based Sampling, to select instances with the Least Confidence.
        sample_pool_pred_prob = classifier.predict_proba(sample_pool)
        lc_indices = self.__query_least_confident(sample_pool_pred_prob, batch_size = 5)


        # for idx, labels_probs in enumerate(sample_pool_pred_prob):
            

        #     if prediction != actual:
        #         # Add to the training set
        #         training_record = X_test.iloc[idx]
        #         self.logger.info(
        #             f'index: {idx}, prediction: {prediction}, actual: {actual}')
        #         new_train = new_train.append(training_record)
        

        return classifier
    
    def __query_least_confident(self, predict_proba_result: [], batch_size: int):
        """Find the instances with the Least Confidence from the result of predict_proba().

        Args:
            predict_proba_result (array): Predicted values from predict_proba().
            batch_size (int): How many instances to add for each Active Learning iteration.
        Returns:
            result (array): List of indices of predict_proba_result with the Least Confidence.
        """
        diff = numpy.diff(predict_proba_result)
        diff = numpy.absolute(diff)        
        diff = diff.flatten()        
        lc_indices = numpy.argpartition(diff, batch_size)
        return lc_indices[:batch_size]

    def learn(self, seed: DataFrame, unlabeled_dataset: DataFrame):
        """Using Scikit-learn and supervised training to create a machine learning model.

        Args:
            seed (DataFrame): Data with the labeled training and test sets.
            unlabeled_dataset: Unlabeled dataset for active learning to select more samples.
        Returns:
            model: a trained machine learning model.
        """

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
        # body_pipeline = Pipeline(
        #     steps=[
        #         ('feature_union', FeatureUnion(
        #             transformer_list=[
        #                 ('body', Pipeline([
        #                     ('selector', DataSubsetSelector(keys='body')),
        #                     ('cv', CountVectorizer(stop_words='english')),
        #                     # ('tfidf', TfidfTransformer()),
        #                 ])),
        #                 # ('pos_features', Pipeline([
        #                 #     ('pos', PosTagEstimator(tokenizer=nltk.word_tokenize)),
        #                 # ])),
        #             ],
        #             transformer_weights={
        #                 'body': 1.0,
        #             },
        #         )),
        #         # Currently SVC has got a better precision/recall and overall accuracy, compared to MultinomialNB.
        #         ('classifier', SVC(kernel='linear'))],
        #     verbose=True)

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

        # TODO separate out categorical_transformer into individual transformer for dialogue_act_classification_ml and comment_is_by_author.
        column_transformer = ColumnTransformer(
            transformers=[
                ('body_bow_pipeline', CountVectorizer(
                    stop_words='english'), 'body'),
                #('body_ngram_pipeline', CountVectorizer(stop_words='english', ngram_range=(1, 3)), 'body'),
                ('categorical_transformer', OneHotEncoder(categories=one_hot_encoder_categories), [
                 'dialogue_act_classification_ml', 'comment_is_by_author']),
                # ('comment_is_by_author_pipeline',
                #  SingleFeatureOneHotEncoder(), 'comment_is_by_author'),
            ],
            transformer_weights={
                'body_bow_pipeline': 1.0,
                # 'body_ngram_pipeline': 0.5,
                'categorical_transformer': 1.0,
                # 'comment_is_by_author_pipeline': 0.1,
            },
            verbose=True)

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", column_transformer),
                ('classifier', SVC(kernel='linear', probability=True))],
            verbose=True)

        # Use Grid Search to perform hyper parameter tuning in order to determine the optimal values for the machine learning model.
        # TODO tweak the search params.
        # {'feature_union__tfdif_features__tfidf__use_idf': (True, False)}
        grid_search_cv_params = {}
        classifier = GridSearchCV(
            full_pipeline, grid_search_cv_params, cv=5, verbose=2)

        # Split data into training and test sets.
        target = seed['code_comprehension_related']
        features = seed[[
            'body', 'dialogue_act_classification_ml', 'comment_is_by_author']]
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            target,
                                                            test_size=0.2,  # 20%
                                                            random_state=2019,  # An arbitrary seed so the results can be reproduced
                                                            stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        # Train the model using the training sets.
        classifier.fit(X_train, y_train)

        # Predict the response for test set.
        y_pred = classifier.predict(X_test)
        y_pred_prob = classifier.predict_proba(X_test)

        # Model accuracy, how often is the classifier correct?
        self.logger.info(
            f'{metrics.classification_report(y_test, y_pred, digits=8)}')

        new_train = X_train
        # Find incorrect predictions
        # Scenario: Pool-Based sampling, batch size = 10.
        # Query Strategy: Least Confidence
        # Small labeled data set is called the seed.
        # Console prompt to ask the Oracle for the label.
        # Stop criteria: Stop at X iteration, or compare with a separate test set to see how the performance has improved or stagnated.
        for idx, prediction in enumerate(y_pred):
            actual = y_test.iloc[idx]

            if prediction != actual:
                # Add to the training set
                training_record = X_test.iloc[idx]
                self.logger.info(
                    f'index: {idx}, prediction: {prediction}, actual: {actual}')
                new_train = new_train.append(training_record)

        # Predict the previously trained YES (code_comprehension_related).
        test_data = [
            ['Should this commented out code still be in here?',
                'ynQuestion', False, 'Yes'],
            ['Can this be private?', 'ynQuestion', False, 'Yes'],
            ['Can it work with "parallel: true"?', 'Emphasis', False, 'Yes'],
            ['I am confused, aren\'t we using `__`?', 'Emphasis', False, 'Yes'],
            ['No need for DatabaseJournalEntry?', 'ynQuestion', False, 'Yes'],
            ['Fixed with #470 ', 'System', True, 'No'],
            ['same comments apply as in python method case', 'Clarify', False, 'No'],
        ]
        test = DataFrame(test_data, columns=[
                         'body', 'dialogue_act_classification_ml', 'comment_is_by_author', 'code_comprehension_related'])
        result_pred = classifier.predict(
            test[['body', 'dialogue_act_classification_ml', 'comment_is_by_author']])
        result_test = test['code_comprehension_related']

        self.logger.info(
            f'{metrics.classification_report(result_test, result_pred, digits=2)}')

        return classifier

        # TODO Stemming
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

        # TODO Lemmatization


# https://github.com/scikit-learn/scikit-learn/issues/12494
#         categories = list(housing.ocean_proximity.unique())

# numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
#                                ("featurization", Feature_Adder_FT),
#                                ("strd_scaler", StandardScaler())], verbose=True)

# categorical_pipeline = Pipeline([("one_hot_encoding", OneHotEncoder(categories=[categories]))],verbose=True)
# # categorical_pipeline = Pipeline([("one_hot_encoding", OneHotEncoder(handle_unknown="ignore"))],verbose=True)

# numerical_features = list(housing.select_dtypes(include=[np.number]).columns)
# categorical_features = list(
#     housing.select_dtypes(include=["category"]).columns)

# full_pipeline = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_features),
#                                    ("categorical_pipeline", categorical_pipeline, categorical_features)],
#                                  verbose=True)

# full_pipeline_with_predictor = Pipeline(
#     [("data_preprocessing", full_pipeline), ("linear_regressor", LinearRegression())])

# param_grid = [
#     {"data_preprocessing__numerical_pipeline__imputer__strategy": ["mean", "median"]}]

# grid_search = GridSearchCV(full_pipeline_with_predictor, param_grid, cv=5, scoring="neg_mean_squared_error",
#                            verbose=2, n_jobs=8)

# grid_search.fit(housing, labels) # works

# grid_search.best_params_
