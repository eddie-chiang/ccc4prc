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

        # Generate unique random numbers.
        random_numbers = random.sample(range(1, total_rows), sample_size)

        data_frame = data_frame.iloc[random_numbers]

        data_frame.to_csv(final_csv, index=False, header=True, mode='w')

        

    #     with open(self.random_row_list_file, mode='r') as random_row_list_file:
    #         random_line_list_reader = reader(random_row_list_file)
    #         random_line_list_2d = list(random_line_list_reader) # 2D array, [row: [value]].
    #         random_line_list = numpy.array(random_line_list_2d).flatten() # Converts 2D array into an 1D array.
    #         random_line_list = list(map(int, random_line_list)) # Converts the list of strings to a list of integers.

    #     self.logger.info(f'There are {len(random_line_list)} samples to be written to {self.output_csv_file}')

    #     with open(self.source_csv_file, mode='r', encoding='utf-8') as input_csvfile:
    #         dict_reader = DictReader(input_csvfile, delimiter=',')

    #         with open(self.output_csv_file, mode='w', newline='', encoding='utf-8') as output_csvfile:
    #             field_names = dict_reader.fieldnames + ['pr_url'] + ['dialogue_act_classification_manual'] + ['dialogue_act_classification_ml']
    #             csv_writer = DictWriter(output_csvfile, field_names, delimiter=',')
    #             csv_writer.writeheader()

    #             sample_rows_generator_expression = (row for idx, row in enumerate(dict_reader) if (idx + 2) in (random_line_list)) # idx + 2 to land on the right rows.

    #             total_samples = len(random_line_list)
    #             ctr = 0
    #             progress_pct = 0

    #             for row in sample_rows_generator_expression:
    #                 # Example: https://api.github.com/repos/hhru/nuts-and-bolts to https://github.com/hhru/nuts-and-bolts/pull/38
    #                 row['pr_url'] = row['project_url'].replace('https://api.github.com/repos', 'https://github.com')
    #                 row['pr_url'] = row['pr_url'] + '/pull/' + row['pullreq_id']
    #                 csv_writer.writerow(row)
    #                 ctr += 1
    #                 progress_pct_floor = math.floor(ctr / total_samples * 100)
    #                 if progress_pct_floor != progress_pct:
    #                     progress_pct = progress_pct_floor
    #                     self.logger.info(f'Progress: {progress_pct}%, sample size written: {ctr}')

    #     self.logger.info(f'Generation completed, output file: {self.output_csv_file}')

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
