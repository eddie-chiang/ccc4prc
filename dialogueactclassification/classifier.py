import nltk

class Classifier:
    def __init__(self, logger):
        self.logger = logger

    # Define a simple feature extractor that checks what words the post contains.
    def dialogue_act_features(self, post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features