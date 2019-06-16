import logging
import math
# import numpy
import pandas
import random
# from csv import DictReader, DictWriter, reader
from pathlib import Path


class FileGenerator:
    """A file generator class for generating a manual labeling file."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self, csv_file: Path):
        """Generate the output csv file.

        Args:
            source_csv_file (Path): A Path object that points to the source .csv file.
        """

        final_csv = Path(csv_file.absolute().as_posix().replace(
            '.csv', '_manual_labelling.csv'))

        data_frame = pandas.read_csv(csv_file)
        total_rows = data_frame.shape[0]
        sample_size = self.__sample_size(total_rows)
        self.logger.info(
            f'No. of rows in {csv_file}: {total_rows}, sample size of 95% confidence level and 5% confidence interval: {sample_size}')

        # Generate unique random numbers start from 0 as iloc has 0 based index.
        random_numbers = random.sample(range(0, total_rows - 1), sample_size)

        data_frame = data_frame.iloc[random_numbers]

        data_frame['comment_url'] = data_frame.apply(
            lambda row: self.__get_comment_url(
                row['project_url'],
                int(row['pullreq_id']),
                int(row['comment_id'])),
            axis=1)

        data_frame.to_csv(final_csv, index=False, header=True, mode='w')
        self.logger.info(f'Generation completed, output file: {final_csv}')

    def __sample_size(self, population_size: int):
        z_score = 1.96  # 95% Confidence Level
        p = 0.5  # 50% Standard of Deviation
        e = 0.05  # 5% Confidence Internval, i.e. margin of error
        N = population_size

        numerator = ((z_score ** 2) * p * (1 - p)) / (e ** 2)
        denominator = 1 + (((z_score ** 2) * p * (1 - p)) / (e ** 2 * N))
        sample_size = numerator / denominator
        sample_size = int(math.ceil(sample_size))

        return sample_size

    def __get_comment_url(self, project_url: str, pullreq_id: int, comment_id: int):
        """Gets the Comment link.
        The format is: https://github.com/{owner}/{repo}/pull/{pullreq_id}#discussion_r{comment_id}
        """

        comment_url = project_url.replace(
            'https://api.github.com/repos', 'https://github.com')
        comment_url = comment_url + '/pull/' + \
            str(pullreq_id) + '#discussion_r' + str(comment_id)
        return comment_url
