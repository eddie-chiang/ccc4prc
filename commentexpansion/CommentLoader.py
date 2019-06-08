import logging
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sshtunnel import SSHTunnelForwarder


class CommentLoader:
    """A data loader class that expands a truncated pull request comment from GHTorrent MongoDB.

    Args:
        ssh_host (str): SSH tunnel host
        ssh_port (int): SSH tunnel port number
        ssh_username (str): SSH tunnel username
        ssh_pkey (str): Path to the SSH private key
        ssh_private_key_password (str): password to the SSH private key
        db_host (str): MongoDB host
        db_port (int): MongoDB port number
        db_username (str): MongoDB username
        db_password (str): MongoDB password
        db (str): MongoDB database
        collection (str): MongoDB collection
    """

    def __init__(self,
                 ssh_host: str,
                 ssh_port: int,
                 ssh_username: str,
                 ssh_pkey: str,
                 ssh_private_key_password: str,
                 db_host: str,
                 db_port: int,
                 db_username: str,
                 db_password: str,
                 db: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.server = SSHTunnelForwarder((ssh_host, ssh_port),
                                         ssh_username=ssh_username,
                                         ssh_pkey=ssh_pkey,
                                         ssh_private_key_password=db_password,
                                         remote_bind_address=(db_host, db_port))

        self.db_username = db_username
        self.db_password = db_password
        self.db = db
        self.collection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.stop()  # Close SSH tunnel

    def __get_connection(self, server: SSHTunnelForwarder):
        if server.is_active:
            return self.collection

        self.logger.info(
            f'SSH Tunnel is not active, connecting to {server.ssh_host}:{server.ssh_port}...')

        while True:
            try:
                server.restart()

                mongo_client = MongoClient('127.0.0.1',
                                           server.local_bind_port,
                                           username=self.db_username,
                                           password=self.db_password,
                                           authSource=self.db,
                                           authMechanism='SCRAM-SHA-1')
                mongo_db = mongo_client[self.db]
                self.collection = mongo_db['pull_request_comments']

                self.logger.info(
                    f'Connecting to MongoDB 127.0.0.1:{server.local_bind_port}.')
                # The ismaster command is cheap and does not require auth.
                mongo_client.admin.command('ismaster')
                self.logger.info('Successfully connected to MongoDB server.')
                return self.collection
            except ConnectionFailure as e:
                self.logger.error(f'MongoDB server is not available, error {e}, retry after 5 seconds.')
                time.sleep(5)

    def load(self, owner: str, repo: str, pullreq_id: int, comment_id: int):
        """Load the full comment.

        Args:
            owner (str): GitHub repository owner.
            repo (str): GitHub repository name.
            pullreq_id (int): Pull request ID.
            comment_id (int): Pull request comment ID.
        """
        query = {"owner": owner,
                 "repo": repo,
                 "pullreq_id": pullreq_id,
                 "id": comment_id}
        collection = self.__get_connection(self.server)
        doc = collection.find_one(query)

        if doc is not None:
            return doc['body']
        else:
            return None
