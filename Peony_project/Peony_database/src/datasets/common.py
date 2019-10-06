import hashlib

from typing import List, Any


def create_hash(hash_args: List[Any]) -> str:
    sha = hashlib.sha256()
    sha.update(" ".join(hash_args).encode())
    return sha.hexdigest()
