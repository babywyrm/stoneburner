from fastapi import Request

from atomics.api.auth import ApiKeyAuth


def _request_with_header(value: str | None) -> Request:
    headers = []
    if value is not None:
        headers.append((b"x-api-key", value.encode()))
    return Request({"type": "http", "headers": headers})


async def test_api_key_auth_valid():
    auth = ApiKeyAuth({"secret-key"})
    assert await auth.authenticate(_request_with_header("secret-key")) is True


async def test_api_key_auth_invalid():
    auth = ApiKeyAuth({"secret-key"})
    assert await auth.authenticate(_request_with_header("wrong-key")) is False


async def test_api_key_auth_missing():
    auth = ApiKeyAuth({"secret-key"})
    assert await auth.authenticate(_request_with_header(None)) is False
