import logging
from pandas import DataFrame

class MachineLearning:
    """A machine learning class for supervised learning to create a model."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def learn(self, data_frame: DataFrame):
        model = None

        self.logger.info(data_frame.describe())

        return model