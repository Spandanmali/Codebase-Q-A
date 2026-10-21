import hashlib


def create_token(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()
