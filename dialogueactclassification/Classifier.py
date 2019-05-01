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
    """
    def __init__(self, logger, trained_classifer_file, train_classifier: bool):
        self.logger = logger

        if trained_classifer_file.is_file() and train_classifier == False:
            with open(trained_classifer_file, mode='rb') as f:
                self.dialogue_act_classifier = pickle.load(f)
                logger.info('Loaded trained dialogue act classifier.')
            
        else:
            logger.info('Training dialogue act classifier.')
            self.dialogue_act_classifier = self.__train()
            
            with open(trained_classifer_file, mode='wb') as f:
                pickle.dump(self.dialogue_act_classifier, f)
                logger.info('Saved trained dialogue act classifier.')   

    def dialogue_act_features(self, post: str):
        """Return a list of words extracted from the given post, as features for the post.

        Args:
            post (str): A sentence, a passage.
        
        Returns:
            A list of tuples ``(contains({word}), True)``.
        """
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features        

    def classify(self, featureset):
        """Classify the given featureset.

        Args:
            featureset: A list of features.

        Returns: 
            The dialogue act classification.
        """
        return self.dialogue_act_classifier.classify(featureset)

    def __train(self):
        # Extract the labeled basic messaging data.
        #posts = nltk.corpus.nps_chat.xml_posts()[:10000]
        posts = nltk.corpus.nps_chat.xml_posts()
        self.logger.info(f'Loaded {len(posts)} posts from nps_chat corpus.')

        # Construct the training and test data by applying the feature extractor to each post, and create a new classifier.
        featuresets = [(self.dialogue_act_features(post.text), post.get('class'))
                        for post in posts]
        size = int(len(featuresets) * 0.1) # 10% to use as Training Set, 90% to use Test Set.
        train_set, test_set = featuresets[size:], featuresets[:size]
        self.logger.info('Size of feature set: %d, train on %d instances, test on %d instances.' % (len(featuresets), len(train_set), len(test_set)))

        # Train the dialogue act classifier.
        dialogue_act_classifier = nltk.NaiveBayesClassifier.train(train_set)
        self.logger.info('Accuracy: {}%.'.format(round(nltk.classify.accuracy(dialogue_act_classifier, test_set) * 100, 4)))

        # Getting the Precision (% of prediction that are correct), and Recall (% of that identifies the class accurately).
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
            self.logger.info(f'Precision ({class_label}): {round(precision_rate * 100, 4)}%.')
            self.logger.info(f'Recall ({class_label}): {round(recall_rate * 100, 4)}%.')

        return dialogue_act_classifier