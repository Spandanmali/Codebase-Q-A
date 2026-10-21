from token import create_token


def login(username: str, password: str) -> str | None:
    if username == "admin" and password == "secret":
        return create_token(username)
    return None
