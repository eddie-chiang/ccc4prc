import csv
import logging
from csv import DictWriter
from google.cloud import bigquery

class CommentsResourceAccess:
    """A resource access for Google BigQuery to get pull request comments from GH Torrent.
    
    Args:
        query_file (Path): A Path object that points to BigQuery .sql file.    
        output_csv_file (Path): A Path object that points to output .csv file.    
    """
    def __init__(self, query_file, output_csv_file):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.query_file = query_file
        self.output_csv_file = output_csv_file

    def export(self):
        """Execute the given BigQuery query, and export the result to a .csv file.

        Args:
            output_csv_file (Path): A Path object that points to the .csv file to export data to.
        """
        with open(self.query_file, 'r') as file:
            my_query = file.read()

        self.logger.info(f"Running BigQuery, query loaded from {self.query_file}: {my_query}")
        client = bigquery.Client()
        query_job = client.query(my_query)  # API request
        rows = query_job.result() # Waits for query to finish

        self.logger.info(f"Running BigQuery, query loaded from {self.query_file}: {my_query}")
        with open(self.output_csv_file, mode='w', newline='', encoding='utf-8') as output_csvfile:      
            field_names = [field.name for field in rows.schema]
            csv_writer = DictWriter(output_csvfile, field_names, delimiter=',')
            csv_writer.writeheader()                               

        for row in rows:
            self.logger.info(f'comment_id: {row.comment_id}, pull_request_id: {row.pull_request_id}, body: {row.body}')

        # TODO
        # 1. Detect language, only take English
        # 2. If the comments is longer than x, then try getting from MongoDB.
        # https://googleapis.github.io/google-cloud-python/latest/bigquery/index.html