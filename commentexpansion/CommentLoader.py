from pymongo import MongoClient
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
        self.server = SSHTunnelForwarder((host, ssh_port),
                                    ssh_username = ssh_username,
                                    ssh_pkey = ssh_pkey,
                                    ssh_private_key_password = ssh_private_key_password,
                                    remote_bind_address=('127.0.0.1', db_port))
        self.server.start()

        self.logger.info(f'Connecting to MongoDB {db_host}:{db_port}')
        self.client = MongoClient(db_host,
                            username = db_username,
                            password = db_password,
                            authSource = db,
                            authMechanism = 'SCRAM-SHA-1')
        self.db = self.client[db]
        self.collection = self.db['pull_request_comments']

    def __enter__(self):
        return self

     def __exit__(self, exc_type, exc_value, traceback):
        # Close ssh tunnel
        self.server.stop()

    def Load(self, owner: str, repo: str, pullreq_id: int, comment_id: int):
        """Load the full comment.

        Args:
            owner (str): GitHub repository owner.
            repo (str): GitHub repository name.
            pullreq_id (int): Pull request ID.
            comment_id (int): Pull request comment ID.
        """

       
