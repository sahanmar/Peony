import pymongo
import hashlib


class MongoDb:
    def __init__(
        self,
        db_user: str = "User",
        db_pass: str = "Pass",
        db_host: str = "127.0.0.1",
        db_port: int = 27017,
    ):

        url = f"mongodb://{db_user}:{db_pass}@{db_host}:{db_port}/Peony-MongoDb"
        self.client = pymongo.MongoClient(url)
        self.databse = self.client["Peony-MongoDb"]


def create_hash(hash_args: list) -> str:
    sha = hashlib.sha256()
    sha.update(" ".join(hash_args).encode())
    return sha.hexdigest()
