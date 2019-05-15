import logging
import math
import numpy
from csv import DictReader, DictWriter, reader
from pathlib import Path

class FileGenerator:
    """A file generator class for generating a manual labeling file.

    Args:
        source_csv_file (Path): A Path object that points to the source .csv file.
        output_csv_file (Path): A Path object that points to the destination output .csv file that will be generated by :func:`~manuallabeling.FileGenerator.generate`.
        random_line_list_file (Path): A Path object that points to a .csv file with a list of random csv row numbers.
    """
    def __init__(self, source_csv_file, output_csv_file, random_row_list_file):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.source_csv_file = source_csv_file
        self.output_csv_file = output_csv_file
        self.random_row_list_file = random_row_list_file
    
    def generate(self):
        """Generate the output csv file."""
            
        with open(self.random_row_list_file, mode='r') as random_row_list_file:
            random_line_list_reader = reader(random_row_list_file)
            random_line_list_2d = list(random_line_list_reader) # 2D array, [row: [value]].
            random_line_list = numpy.array(random_line_list_2d).flatten() # Converts 2D array into an 1D array.
            random_line_list = list(map(int, random_line_list)) # Converts the list of strings to a list of integers.

        self.logger.info(f'There are {len(random_line_list)} samples to be written to {self.output_csv_file}')

        with open(self.source_csv_file, mode='r', encoding='utf-8') as input_csvfile:
            dict_reader = DictReader(input_csvfile, delimiter=',')

            with open(self.output_csv_file, mode='w', newline='', encoding='utf-8') as output_csvfile:
                field_names = dict_reader.fieldnames + ['dialogue_act_classification_manual'] + ['dialogue_act_classification_ml']
                csv_writer = DictWriter(output_csvfile, field_names, delimiter=',')
                csv_writer.writeheader()

                sample_rows_generator_expression = (row for idx, row in enumerate(dict_reader) if (idx + 2) in (random_line_list)) # idx + 2 to land on the right rows.

                total_samples = len(random_line_list)
                ctr = 0
                progress_pct = 0

                for row in sample_rows_generator_expression: 
                    csv_writer.writerow(row)
                    ctr += 1
                    progress_pct_floor = math.floor(ctr / total_samples * 100)
                    if progress_pct_floor != progress_pct:
                        progress_pct = progress_pct_floor
                        self.logger.info(f'Progress: {progress_pct}%, sample size written: {ctr}')
                    
        self.logger.info(f'Generation completed, output file: {self.output_csv_file}')