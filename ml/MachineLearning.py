import logging
from array import array
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot
from pandas import DataFrame
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


class MachineLearning:
    """A machine learning class for supervised learning to create a model."""

    FEATURES = ['body', 'dialogue_act_classification_ml', 'comment_is_by_author']
    LABEL = 'code_comprehension_related'

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def active_learn(self, training_dataset: DataFrame, training_dataset_file: Path, test_dataset: DataFrame, unlabeled_dataset: DataFrame):
        """Using Scikit-learn, supervised training and active learning to create a machine learning model.

        Args:
            training_dataset (DataFrame): Training dataset, with existing labeled instances.
            training_dataset_file (Path): Training dataset file path, for the new labeled instances from active learning to append to.
            test_dataset (DataFrame): Separate test dataset to measure the performance of the machine learning model.
            unlabeled_dataset (DataFrame): Pool of unlabeled data for pool-based sampling to select new instances from. This may contain instances from training or test datasets, which would be filtered out when querying new instances to send to Oracle to label.
        Returns:
            model: a trained machine learning model.
        """

        classifier = self.__get_classifier()

        X_train = training_dataset[self.FEATURES]
        X_test = test_dataset[self.FEATURES]
        y_train = training_dataset[self.LABEL]
        y_true = test_dataset[self.LABEL]

        classifier, report, report_dict = self.__train_model(classifier, X_train, X_test, y_train, y_true)
        self.logger.info(f'{report}')

        report_list = [report]
        report_dict_list = [report_dict]

        self.logger.info('......Active Learning Starts......')
        n_instances = 0

        while True:
            self.logger.info(f'Scenario: Pool-based sampling...')
            # Draw instances from the unlabelled dataset, filter out already labeled instances in training and test datasets.
            comment_ids = pandas.concat([training_dataset['comment_id'], test_dataset['comment_id']])
            unlabeled_dataset = unlabeled_dataset[~unlabeled_dataset.comment_id.isin(comment_ids)]
            pool = unlabeled_dataset[self.FEATURES]

            self.logger.info(f'Query Strategy: Least Confidence...')
            pool_pred_prob = classifier.predict_proba(pool)
            idx = self.__query_least_confident(pool_pred_prob, batch_size=1)[0]
            instance = unlabeled_dataset.iloc[[idx]].to_dict('records')[0]
            instance_prob = pool_pred_prob[idx]
            n_instances += 1

            self.logger.info(
                f'Instance No. {n_instances}, label ({classifier.classes_[0]}): {instance_prob[0]:.2%}, label ({classifier.classes_[1]}): {instance_prob[1]:.2%}, comment_id: {instance["comment_id"]}')

            code_comprehension_related, topic_keywords, problem_encountered = self.__label_instance_by_oracle(
                instance['body'], instance['dialogue_act_classification_ml'], instance['comment_is_by_author'])

            instance['code_comprehension_related'] = code_comprehension_related
            instance['problem_encountered'] = problem_encountered
            instance['topic_keywords'] = topic_keywords

            # unlabeled_dataset instance has less columns than training_dataset, so use pandas.concat.
            training_dataset = pandas.concat([training_dataset, DataFrame(instance, index=[0])], ignore_index=True)

            # Append the last row (the newly labeled instance) to file.
            # Use -1, as pandas.to_csv() appends a newline at the EOF.
            training_dataset.iloc[[-1]].to_csv(training_dataset_file, header=False, index=False, mode='a')

            # Retrain the model with newly labeled instance.
            X_train = training_dataset[self.FEATURES]
            y_train = training_dataset[self.LABEL]
            classifier, report, report_dict = self.__train_model(
                classifier, X_train, X_test, y_train, y_true)

            report_list.append(report)
            report_dict_list.append(report_dict)

            # Report the classifier performance, and ask the user to determine whether the stopping criteria is met.
            if self.__test_stopping_criteria(n_instances, report_list, report_dict_list):
                return classifier

    def __train_model(self, classifier, X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_true: DataFrame):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Model accuracy, how often is the classifier correct?
        report = metrics.classification_report(y_true, y_pred, digits=8)
        report_dict = metrics.classification_report(y_true, y_pred, output_dict=True)

        return classifier, report, report_dict

    def __get_classifier(self):
        """ Construct and return a machine learning classifier.
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
            verbose=False)

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", column_transformer),
                ('classifier', SVC(kernel='linear', probability=True))],
            verbose=False)

        # Use Grid Search to perform hyper parameter tuning in order to determine the optimal values for the machine learning model.
        # TODO tweak the search params.
        # {'feature_union__tfdif_features__tfidf__use_idf': (True, False)}
        grid_search_cv_params = {}
        classifier = GridSearchCV(
            full_pipeline, grid_search_cv_params, cv=5, verbose=0)

        return classifier

    def __query_least_confident(self, predict_proba_result: array, batch_size: int = 1):
        """Find the instances with the Least Confidence from the result of predict_proba().

        Args:
            predict_proba_result (array): Predicted values from predict_proba().
            batch_size (int): How many instances to add for each Active Learning iteration. Defaults to 1.
        Returns:
            array: List of indices of predict_proba_result with the Least Confidence.
        """
        diff = numpy.diff(predict_proba_result)
        diff = numpy.absolute(diff)
        diff = diff.flatten()
        lc_indices = numpy.argpartition(diff, batch_size)
        return lc_indices[:batch_size]

    def __label_instance_by_oracle(self, body: str, dialogue_act_classification_ml: str, comment_is_by_author: bool):
        """Send the query instance to the oracle (i.e. human annotator) to label.

        Args:
            body (str): Review comment body.
            dialogue_act_classification_ml (str): Dialogue act classification derived from Machine Learning.
            comment_is_by_author (bool): Is the comment by the pull request author?
        Returns:
            (str, str, str): A tuple comprising the annotated result from the oracle: code_comprehension_related label, topic_keywords, and problem_encountered.
        """
        print(f"body: {body}")
        print(f"dialogue_act_classification_ml: {dialogue_act_classification_ml}")
        print(f"comment_is_by_author: {comment_is_by_author}")

        while True:
            oracle_input = input('Code Comprehension Related? (y/n) ').lower()
            if oracle_input == 'y' or oracle_input == 'n':
                code_comprehension_related = 'Yes' if oracle_input == 'y' or oracle_input == 'yes' else 'No'
                break

        topic_keywords = input('Topic keywords (semicolon separated)? ')

        problem_encountered = ''
        if code_comprehension_related == 'Yes':
            problem_encountered_type_1 = "What is the program supposed to do"
            problem_encountered_type_2 = "What was the developer's intention when writing this code"
            problem_encountered_type_3 = "Why was this code implemented this way"
            problem_encountered_type_4 = "Who has experience with this code"

            print('Type of problem encountered?')
            print(f'1. {problem_encountered_type_1}?')
            print(f'2. {problem_encountered_type_2}?')
            print(f'3. {problem_encountered_type_3}?')
            print(f'4. {problem_encountered_type_4}?')

            while True:
                oracle_input = input('Choice (1-4): ')
                if oracle_input.isnumeric():
                    oracle_input_int = int(oracle_input)
                    if oracle_input_int == 1:
                        problem_encountered = problem_encountered_type_1
                        break
                    elif oracle_input_int == 2:
                        problem_encountered = problem_encountered_type_2
                        break
                    elif oracle_input_int == 3:
                        problem_encountered = problem_encountered_type_3
                        break
                    elif oracle_input_int == 4:
                        problem_encountered = problem_encountered_type_4
                        break

        print()
        return code_comprehension_related, topic_keywords, problem_encountered

    def __test_stopping_criteria(self, n_instances: int, report_list: array, report_dict_list: array):
        """Display the classification report comparison to the user, showing the difference in performance before and after the Active Learning iteration.
        Then prompt the user whether to continue or stop Active Learning.

        Args:
            n_instances (int): Number of instances active learning queried.
            report_list (array): A history of classification reports text after the active learning iteration.
            report_dict_list (array): A history of classification reports of the accuracy, recal and precision rates.

        Returns:
            bool: true - stop active learning, false - continue active learning.
        """
        self.logger.info(f'Active Learning "{n_instances}" instances queried.')
        self.logger.info(f'{report_list[-1]}')

        x = range(len(report_dict_list))
        df = DataFrame({
            'x': x,
            'no_precision': [i['No']['precision'] for i in report_dict_list],
            'no_recall': [i['No']['recall'] for i in report_dict_list],
            'no_f1_score': [i['No']['f1-score'] for i in report_dict_list],
            'yes_precision': [i['Yes']['precision'] for i in report_dict_list],
            'yes_recall': [i['Yes']['recall'] for i in report_dict_list],
            'yes_f1_score': [i['Yes']['f1-score'] for i in report_dict_list],
            'Accuracy': [i['accuracy'] for i in report_dict_list]})

        with pyplot.style.context('seaborn-white'):
            _, axs = pyplot.subplots(1, 3)
            pyplot.setp(axs, xticks=x, xlabel='Number of Query Instances')

            axs[0].set_title('Label "No"')
            axs[0].plot('x', 'no_precision', data=df, marker='o', color='red', label='Precision')
            axs[0].plot('x', 'no_recall', data=df, marker='+', color='orange', label='Recall')
            axs[0].plot('x', 'no_f1_score', data=df, marker='*', color='blue', label='F1 Score')
            axs[0].legend()

            axs[1].set_title('Label "Yes"')
            axs[1].plot('x', 'yes_precision', data=df, marker='o', color='red', label='Precision')
            axs[1].plot('x', 'yes_recall', data=df, marker='+', color='orange', label='Recall')
            axs[1].plot('x', 'yes_f1_score', data=df, marker='*', color='blue', label='F1 Score')
            axs[1].legend()

            axs[2].set_title('Accuracy')
            axs[2].plot('x', 'Accuracy', data=df, marker='o', color='red', linewidth=2)

            pyplot.show(block=False)

        while True:
            choice = input('Continue Active Learning? (y/n) ').lower()
            if choice == 'y' or choice == 'n':
                return choice == 'n'

    def train_test_split(self, data: DataFrame):
        """Split data into training and test datasets.

        Args:
            data (DataFrame): Sample dataset with the labeled data.

        Returns:
            tuple: (
                DataFrame: Training dataset.
                DataFrame: Test dataset.
            )
        """
        target = data[self.LABEL]
        features = data[self.FEATURES]

        X_train, X_test, _, _ = train_test_split(
            features,
            target,
            test_size=0.2,  # 20%
            random_state=2019,  # An arbitrary seed so the results can be reproduced
            stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

        training_dataset = data.iloc[X_train.index]
        test_dataset = data.iloc[X_test.index]

        return training_dataset, test_dataset

    # def learn(self, seed: DataFrame, unlabeled_dataset: DataFrame):
    #     """Using Scikit-learn and supervised training to create a machine learning model.

    #     Args:
    #         seed (DataFrame): Data with the labeled training and test sets.
    #         unlabeled_dataset: Unlabeled dataset for active learning to select more samples.
    #     Returns:
    #         model: a trained machine learning model.
    #     """

    #     # # Create a Gaussian Naive Bayes classifier.
    #     # model = GaussianNB()

    #     # # Declare data preprocessing steps.
    #     # # Encode features, i.e. convert string labels into numbers.
    #     # label_encoder = preprocessing.LabelEncoder()

    #     # dialogue_act_classification_ml = data_frame['dialogue_act_classification_ml']
    #     # dialogue_act_classification_ml_encoded = label_encoder.fit_transform(
    #     #     dialogue_act_classification_ml)

    #     # comment_is_by_author = data_frame['comment_is_by_author']
    #     # comment_is_by_author_encoded = label_encoder.fit_transform(
    #     #     comment_is_by_author)

    #     # # body = data_frame['body']
    #     # # count_vectorizer = CountVectorizer(stop_words='english')

    #     # # Combinig features into single listof tuples
    #     # features = list(zip(dialogue_act_classification_ml_encoded,
    #     #                     comment_is_by_author_encoded))

    #     # # Split data into training and test sets.
    #     # target = data_frame['code_comprehension_related']
    #     # X_train, X_test, y_train, y_test = train_test_split(features,
    #     #                                                     target,
    #     #                                                     test_size=0.2,  # 20%
    #     #                                                     random_state=2019,  # An arbitrary seed so the results can be reproduced
    #     #                                                     stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

    #     # # Train the model using the training sets.
    #     # model.fit(X_train, y_train)

    #     # # Predict the response for test set.
    #     # y_pred = model.predict(X_test)

    #     # # Model accuracy, how often is the classifier correct?
    #     # self.logger.info(f'{metrics.classification_report(y_test, y_pred, digits=8)}')

    #     # return model

    #     # # Create a Multinomial Naive Bayes classifier.
    #     # # Make a pipeline for data preprocessing.
    #     # # Encode features, i.e. convert string labels into numbers.
    #     # classifier = make_pipeline(preprocessing.OneHotEncoder(), MultinomialNB())

    #     # # Split data into training and test sets.
    #     # target = data_frame['code_comprehension_related']
    #     # features = data_frame[['dialogue_act_classification_ml', 'comment_is_by_author']]
    #     # X_train, X_test, y_train, y_test = train_test_split(features,
    #     #                                                     target,
    #     #                                                     test_size=0.2,  # 20%
    #     #                                                     random_state=2019,  # An arbitrary seed so the results can be reproduced
    #     #                                                     stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

    #     # # Train the model using the training sets.
    #     # classifier.fit(X_train, y_train)

    #     # # Predict the response for test set.
    #     # y_pred = classifier.predict(X_test)

    #     # # Model accuracy, how often is the classifier correct?
    #     # self.logger.info(f'{metrics.classification_report(y_test, y_pred, digits=8)}')

    #     # return classifier

    #     # Create a Multinomial Naive Bayes classifier.
    #     # Make a pipeline for data preprocessing.
    #     # Encode features, i.e. convert string labels into numbers.
    #     # body_pipeline = Pipeline(
    #     #     steps=[
    #     #         ('feature_union', FeatureUnion(
    #     #             transformer_list=[
    #     #                 ('body', Pipeline([
    #     #                     ('selector', DataSubsetSelector(keys='body')),
    #     #                     ('cv', CountVectorizer(stop_words='english')),
    #     #                     # ('tfidf', TfidfTransformer()),
    #     #                 ])),
    #     #                 # ('pos_features', Pipeline([
    #     #                 #     ('pos', PosTagEstimator(tokenizer=nltk.word_tokenize)),
    #     #                 # ])),
    #     #             ],
    #     #             transformer_weights={
    #     #                 'body': 1.0,
    #     #             },
    #     #         )),
    #     #         # Currently SVC has got a better precision/recall and overall accuracy, compared to MultinomialNB.
    #     #         ('classifier', SVC(kernel='linear'))],
    #     #     verbose=True)

    #     one_hot_encoder_categories = [
    #         [
    #             'Accept',
    #             'Bye',
    #             'Clarify',
    #             'Continuer',
    #             'Emotion',
    #             'Emphasis',
    #             'Greet',
    #             'Other',
    #             'Reject',
    #             'Statement',
    #             'System',
    #             'whQuestion',
    #             'yAnswer',
    #             'nAnswer',
    #             'ynQuestion'
    #         ],
    #         [
    #             False,  # 0 should come before 1 for numerical columns.
    #             True
    #         ]
    #     ]

    #     # TODO separate out categorical_transformer into individual transformer for dialogue_act_classification_ml and comment_is_by_author.
    #     column_transformer = ColumnTransformer(
    #         transformers=[
    #             ('body_bow_pipeline', CountVectorizer(
    #                 stop_words='english'), 'body'),
    #             #('body_ngram_pipeline', CountVectorizer(stop_words='english', ngram_range=(1, 3)), 'body'),
    #             ('categorical_transformer', OneHotEncoder(categories=one_hot_encoder_categories), [
    #              'dialogue_act_classification_ml', 'comment_is_by_author']),
    #             # ('comment_is_by_author_pipeline',
    #             #  SingleFeatureOneHotEncoder(), 'comment_is_by_author'),
    #         ],
    #         transformer_weights={
    #             'body_bow_pipeline': 1.0,
    #             # 'body_ngram_pipeline': 0.5,
    #             'categorical_transformer': 1.0,
    #             # 'comment_is_by_author_pipeline': 0.1,
    #         },
    #         verbose=True)

    #     full_pipeline = Pipeline(
    #         steps=[
    #             ("preprocessor", column_transformer),
    #             ('classifier', SVC(kernel='linear', probability=True))],
    #         verbose=True)

    #     # Use Grid Search to perform hyper parameter tuning in order to determine the optimal values for the machine learning model.
    #     # TODO tweak the search params.
    #     # {'feature_union__tfdif_features__tfidf__use_idf': (True, False)}
    #     grid_search_cv_params = {}
    #     classifier = GridSearchCV(
    #         full_pipeline, grid_search_cv_params, cv=5, verbose=2)

    #     # Split data into training and test sets.
    #     target = seed['code_comprehension_related']
    #     features = seed[[
    #         'body', 'dialogue_act_classification_ml', 'comment_is_by_author']]
    #     X_train, X_test, y_train, y_test = train_test_split(features,
    #                                                         target,
    #                                                         test_size=0.2,  # 20%
    #                                                         random_state=2019,  # An arbitrary seed so the results can be reproduced
    #                                                         stratify=target)  # Stratify the sample by the target (i.e. code_comprehension_related)

    #     # Train the model using the training sets.
    #     classifier.fit(X_train, y_train)

    #     # Predict the response for test set.
    #     y_pred = classifier.predict(X_test)
    #     y_pred_prob = classifier.predict_proba(X_test)

    #     # Model accuracy, how often is the classifier correct?
    #     self.logger.info(
    #         f'{metrics.classification_report(y_test, y_pred, digits=8)}')

    #     new_train = X_train
    #     # Find incorrect predictions
    #     # Scenario: Pool-Based sampling, batch size = 10.
    #     # Query Strategy: Least Confidence
    #     # Small labeled data set is called the seed.
    #     # Console prompt to ask the Oracle for the label.
    #     # Stop criteria: Stop at X iteration, or compare with a separate test set to see how the performance has improved or stagnated.
    #     for idx, prediction in enumerate(y_pred):
    #         actual = y_test.iloc[idx]

    #         if prediction != actual:
    #             # Add to the training set
    #             training_record = X_test.iloc[idx]
    #             self.logger.info(
    #                 f'index: {idx}, prediction: {prediction}, actual: {actual}')
    #             new_train = new_train.append(training_record)

    #     # Predict the previously trained YES (code_comprehension_related).
    #     test_data = [
    #         ['Should this commented out code still be in here?',
    #             'ynQuestion', False, 'Yes'],
    #         ['Can this be private?', 'ynQuestion', False, 'Yes'],
    #         ['Can it work with "parallel: true"?', 'Emphasis', False, 'Yes'],
    #         ['I am confused, aren\'t we using `__`?', 'Emphasis', False, 'Yes'],
    #         ['No need for DatabaseJournalEntry?', 'ynQuestion', False, 'Yes'],
    #         ['Fixed with #470 ', 'System', True, 'No'],
    #         ['same comments apply as in python method case', 'Clarify', False, 'No'],
    #     ]
    #     test = DataFrame(test_data, columns=[
    #                      'body', 'dialogue_act_classification_ml', 'comment_is_by_author', 'code_comprehension_related'])
    #     result_pred = classifier.predict(
    #         test[['body', 'dialogue_act_classification_ml', 'comment_is_by_author']])
    #     result_test = test['code_comprehension_related']

    #     self.logger.info(
    #         f'{metrics.classification_report(result_test, result_pred, digits=2)}')

    #     return classifier

    #     # TODO Stemming
    #     # Example code
    #     # Stemming Code
    #     # from nltk.stem.snowball import SnowballStemmer
    #     # stemmer = SnowballStemmer("english", ignore_stopwords=True)

    #     # class StemmedCountVectorizer(CountVectorizer):
    #     #     def build_analyzer(self):
    #     #         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
    #     #         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    #     # stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

    #     # text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
    #     #                             ('mnb', MultinomialNB(fit_prior=False))])

    #     # text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)

    #     # predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

    #     # np.mean(predicted_mnb_stemmed == twenty_test.target)

    #     # TODO Lemmatization


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
