import collections
import nltk
import pickle
from nltk.metrics.scores import precision, recall

class Classifier:
    """A classifier for dialogue act classification.

    Args:
        logger (Logger): A logger.
        trained_classifier_file (Path): A Path object that points to the trained classifier .pickle file.
        train_classifier (bool): If the trained classifier .pickle file already exists, whether to retrain the classifier.
            If the .pickle file does not exist, then a new dialogue act classifier will be trained, and save to trained_classifier_file.
        test_set_percentage (int): The percentage of labeled NPS Chat corpus to be used as the test set (remainder will be used as the train set).
    """
    def __init__(self, logger, trained_classifer_file, train_classifier: bool, test_set_percentage: int):
        self.logger = logger

        test_set = None
        if trained_classifer_file.is_file() and train_classifier == False:
            with open(trained_classifer_file, mode='rb') as f:
                self.dialogue_act_classifier = pickle.load(f)
                logger.info('Loaded trained dialogue act classifier.')
            _, _, test_set = self.__get_featuresets(test_set_percentage)            
        else:
            logger.info('Training dialogue act classifier.')
            self.dialogue_act_classifier, test_set = self.__train(test_set_percentage)
            
            with open(trained_classifer_file, mode='wb') as f:
                pickle.dump(self.dialogue_act_classifier, f)
                logger.info('Saved trained dialogue act classifier.')   

        self.__log_accuracy_precision_recall(self.dialogue_act_classifier, test_set)

    def classify(self, dialgoue: str):
        """Classify the given featureset.

        Args:
            dialogue (str): A sentence, a passage.

        Returns: 
            str: The dialogue act type.
        """
        unlabeled_data_features = self.__dialogue_act_features(dialgoue)
        return self.dialogue_act_classifier.classify(unlabeled_data_features)

    def __dialogue_act_features(self, dialogue: str):
        features = {}
        for word in nltk.word_tokenize(dialogue):
            features['contains({})'.format(word.lower())] = True
        return features        

    def __train(self, test_set_percentage: int):
        featuresets, train_set, test_set = self.__get_featuresets(test_set_percentage)
        self.logger.info('Size of feature set: %d, train on %d instances, test on %d instances.' % (len(featuresets), len(train_set), len(test_set)))

        # Train the dialogue act classifier.
        return nltk.NaiveBayesClassifier.train(train_set), test_set

    def __log_accuracy_precision_recall(self, dialogue_act_classifier, test_set: []):
        self.logger.info('Accuracy: {}%.'.format(round(nltk.classify.accuracy(dialogue_act_classifier, test_set) * 100, 4)))

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (features, class_label) in enumerate(test_set):
            refsets[class_label].add(i)
            observed = dialogue_act_classifier.classify(features)
            testsets[observed].add(i)

        for class_label in refsets:
            precision_rate = precision(refsets[class_label], testsets[class_label])
            recall_rate = recall(refsets[class_label], testsets[class_label])
            if not isinstance(precision_rate, float):
                precision_rate = 0
            if not isinstance(recall_rate, float):
                recall_rate = 0
            self.logger.info(f'Precision ({class_label}): {round(precision_rate * 100, 4)}%.') # % of prediction that are correct
            self.logger.info(f'Recall ({class_label}): {round(recall_rate * 100, 4)}%.') # % of that identifies the class accurately

    def __get_featuresets(self, test_set_percentage: int):
        # Extract the labeled basic messaging data.
        posts = nltk.corpus.nps_chat.xml_posts()

        # Construct the train and test data by applying the feature extractor to each post, and create a new classifier.
        featuresets = [(self.__dialogue_act_features(post.text), post.get('class'))
                        for post in posts]
        test_set_size = int(len(featuresets) * test_set_percentage / 100)
        train_set, test_set = featuresets[test_set_size:], featuresets[:test_set_size]

        return featuresets, train_set, test_set