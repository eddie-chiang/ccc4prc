import collections
import logging
import pickle
from pathlib import Path

from nltk import NaiveBayesClassifier, corpus, word_tokenize
from nltk.classify import accuracy
from nltk.metrics.scores import precision, recall
from pandas import read_csv
from tqdm import tqdm


class DialogueActClassifierFactory:
    """Factory to create a classifier for dialogue act classification.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.clf = None

    def get_classifier(self, classifier_file: Path, test_set_percentage: int) -> NaiveBayesClassifier:
        """Train the classifier and persist the model to the specified file, or load from an existing model.

        Args:
            classifier_file (Path): A Path object that points to the trained classifier .pickle file.
            test_set_percentage (int): The percentage of labeled NPS Chat corpus to be used as the test set (remainder will be used as the train set).

        Returns:
            NaiveBayesClassifier: Trained classifier.
        """
        if self.clf != None:
            return self.clf

        if classifier_file.is_file():
            with open(classifier_file, mode='rb') as f:
                self.clf = pickle.load(f)
                self.logger.info('Loaded trained dialogue act classifier.')
            _, _, self.test_set = self.__get_featuresets(test_set_percentage)
        else:
            self.logger.info('Training dialogue act classifier.')
            self.clf, self.test_set = self.__train(test_set_percentage)

            with open(classifier_file, mode='wb') as f:
                pickle.dump(self.clf, f)
                self.logger.info('Saved trained dialogue act classifier.')

        return self.clf

    def classify(self, dialogue: str) -> str:
        """Classify the given featureset.

        Args:
            dialogue (str): A sentence, a passage.

        Returns: 
            str: The dialogue act type.
        """
        unlabeled_data_features = self.__dialogue_act_features(dialogue)
        return self.clf.classify(unlabeled_data_features)

    def classify_prc_csv_file(self, prc_csv_file: Path) -> Path:
        """Classify the given Pull Request Comments .csv file.

        Args:
            prc_csv_file (Path): A Path object that points to the Pull Request Comments .csv file.

        Returns:
            Path: The file path of the output file.
        """
        classified_csv_file = Path(prc_csv_file.absolute().as_posix().replace('.csv', '_dac_classified.csv'))

        if classified_csv_file.exists():
            self.logger.info(f'Output file already exists, stop further processing: {classified_csv_file}')
            return classified_csv_file

        data_frame = read_csv(prc_csv_file)
        tqdm.pandas(desc='Classifying Dialogue Act')
        data_frame['dialogue_act_classification_ml'] = data_frame.progress_apply(
            lambda row: self.classify(row['body']),
            axis='columns'
        )
        data_frame.to_csv(classified_csv_file, index=False, header=True, mode='w')
        self.logger.info(f'Dialogue Act Classification completed, output file: {classified_csv_file}')
        self.__classification_report()

        return classified_csv_file

    def __dialogue_act_features(self, dialogue: str) -> dict:
        features = {}
        for word in word_tokenize(dialogue):
            features['contains({})'.format(word.lower())] = True
        return features

    def __train(self, test_set_percentage: int):
        featuresets, train_set, test_set = self.__get_featuresets(test_set_percentage)
        self.logger.info(
            f'Size of feature set: {len(featuresets)}, train on {len(train_set)} instances, test on {len(test_set)} instances.')

        # Train the dialogue act classifier.
        return NaiveBayesClassifier.train(train_set), test_set

    def __classification_report(self):
        """Prints classifier accuracy, precisions and recalls.
        """
        self.logger.info(f'Accuracy: {self.get_accuracy()}')

        precisions, recalls = self.get_precision_and_recall()
        for label in precisions.keys():
            self.logger.info(f'{label} - precision: {precisions[label]}, recall: {recalls[label]}')

    def get_accuracy(self):
        """Returns the Accuracy of the Dialogue Act Classifier.

        Returns:
            float: Accuracy.
        """
        return accuracy(self.clf, self.test_set)

    def get_precision_and_recall(self):
        """Returns the Precision and Recall for each class label of the Dialogue Act Classifier.

        Returns:
            tuple: (
                dict: A dictionary of the class labels and the corresponding precision.
                dict: A dictionary of the class labels and the corresponding recall.
            )
        """
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        precisions = dict()
        recalls = dict()

        for i, (features, class_label) in enumerate(self.test_set):
            refsets[class_label].add(i)
            observed = self.clf.classify(features)
            testsets[observed].add(i)

        for class_label in refsets:
            precisions[class_label] = precision(refsets[class_label], testsets[class_label])
            recalls[class_label] = recall(refsets[class_label], testsets[class_label])

        return precisions, recalls

    def __get_featuresets(self, test_set_percentage: int):
        # Extract the labeled basic messaging data.
        posts = corpus.nps_chat.xml_posts()

        # Construct the train and test data by applying the feature extractor to each post, and create a new classifier.
        featuresets = [(self.__dialogue_act_features(post.text), post.get('class'))
                       for post in posts]
        test_set_size = int(len(featuresets) * test_set_percentage / 100)
        train_set, test_set = featuresets[test_set_size:], featuresets[:test_set_size]

        return featuresets, train_set, test_set
