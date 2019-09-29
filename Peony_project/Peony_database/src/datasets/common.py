import hashlib

from typing import List


def create_hash(hash_args: List[any]) -> str:
    sha = hashlib.sha256()
    sha.update(" ".join(hash_args).encode())
    return sha.hexdigest()
