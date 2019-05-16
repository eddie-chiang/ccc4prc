import logging
from google.cloud import bigquery

class CommentsResourceAccess:
    """A resource access for Google BigQuery to get pull request comments from GH Torrent.

    https://googleapis.github.io/google-cloud-python/latest/bigquery/index.html
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def export(self):
        """Execute the given BigQuery query, and export the result to a .csv file.

        Args:
            output_csv_file (Path): A Path object that points to the .csv file to export data to.
        """
        client = bigquery.Client()

        # Perform a query.
        QUERY = (
            'select comment_id, body, pull_request_id from `ghtorrent-bq.ght_2018_04_01.pull_request_comments` where pull_request_id = 19352525')
        query_job = client.query(QUERY)  # API request
        rows = query_job.result()  # Waits for query to finish

        for row in rows:
            self.logger.info(f'comment_id: {row.comment_id}, pull_request_id: {row.pull_request_id}, body: {row.body}')

        # TODO
        # 1. Detect language, only take English
        # 2. If the comments is longer than x, then try getting from MongoDB.