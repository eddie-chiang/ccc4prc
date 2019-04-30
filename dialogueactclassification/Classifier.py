import collections
import nltk
from nltk.metrics.scores import precision, recall

class Classifier:
    def __init__(self, logger):
        self.logger = logger

    # Define a simple feature extractor that checks what words the post contains.
    def dialogue_act_features(self, post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features
    
    def train(self):
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