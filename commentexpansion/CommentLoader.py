import logging
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

        # Creates an SSH tunnel
        self.logger.info(f'Creating an SSH tunnel to {ssh_host}:{ssh_port}')
        self.server = SSHTunnelForwarder((ssh_host, ssh_port),
                                    ssh_username = ssh_username,
                                    ssh_pkey = ssh_pkey,
                                    ssh_private_key_password = db_password,
                                    remote_bind_address=(db_host, db_port))
        self.server.start()
        
        client = MongoClient('127.0.0.1',
                            self.server.local_bind_port,
                            username = db_username,
                            password = db_password,
                            authSource = db,
                            authMechanism = 'SCRAM-SHA-1')
        self.db = client[db]
        self.collection = self.db['pull_request_comments']

        try:
            self.logger.info(f'Connecting to MongoDB 127.0.0.1:{self.server.local_bind_port}.')
            client.admin.command('ismaster') # The ismaster command is cheap and does not require auth.
            self.logger.info('Successfully connected to MongoDB server.')
        except ConnectionFailure:
            self.logger.errors('MongoDB server is not available.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.stop() # Close SSH tunnel

    def Load(self, owner: str, repo: str, pullreq_id: int, comment_id: int):
        """Load the full comment.

        Args:
            owner (str): GitHub repository owner.
            repo (str): GitHub repository name.
            pullreq_id (int): Pull request ID.
            comment_id (int): Pull request comment ID.
        """
        query = { 
            "owner": owner,
            "repo": repo,
            "pullreq_id": pullreq_id,
            "id": comment_id }
        
        mydoc = self.collection.find(query)

        for x in mydoc:
            self.logger.info(x)
       
