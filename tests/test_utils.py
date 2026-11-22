from contextlib import nullcontext

import pytest

from contextlib import AbstractContextManager

from jwt.utils import force_bytes, from_base64url_uint, is_ssh_key, to_base64url_uint


@pytest.mark.parametrize(
    "inputval,expected",
    [
        (0, nullcontext(b"AA")),
        (1, nullcontext(b"AQ")),
        (255, nullcontext(b"_w")),
        (65537, nullcontext(b"AQAB")),
        (123456789, nullcontext(b"B1vNFQ")),
        (-1, pytest.raises(ValueError)),
    ],
)
def test_to_base64url_uint(
    inputval: int, expected: AbstractContextManager[bytes]
) -> None:
    with expected as e:
        actual = to_base64url_uint(inputval)
        assert actual == e


@pytest.mark.parametrize(
    "inputval,expected",
    [
        (b"AA", 0),
        (b"AQ", 1),
        (b"_w", 255),
        (b"AQAB", 65537),
        (b"B1vNFQ", 123456789),
    ],
)
def test_from_base64url_uint(inputval: bytes, expected: int) -> None:
    actual = from_base64url_uint(inputval)
    assert actual == expected


def test_force_bytes_raises_error_on_invalid_object() -> None:
    with pytest.raises(TypeError):
        force_bytes({})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key_format",
    (
        b"ssh-ed25519",
        b"ssh-rsa",
        b"ssh-dss",
        b"ecdsa-sha2-nistp256",
        b"ecdsa-sha2-nistp384",
        b"ecdsa-sha2-nistp521",
        b"sk-ssh-ed25519@openssh.com",
        b"sk-ecdsa-sha2-nistp256@openssh.com",
        b"ssh-rsa-cert-v01@openssh.com",
        b"ssh-ed25519-cert-v01@openssh.com",
        b"sk-ssh-ed25519-cert-v01@openssh.com",
    ),
)
def test_is_ssh_key(key_format: bytes) -> None:
    blob = b"A" * 70
    assert is_ssh_key(key_format + b" " + blob) is True
    assert is_ssh_key(key_format + b" " + blob + b" a comment") is True
    assert is_ssh_key(b"\n " + key_format + b" " + blob) is True
    assert is_ssh_key(b"no-port-forwarding " + key_format + b" " + blob) is True
    assert is_ssh_key(b"@cert-authority " + key_format + b" " + blob) is True
    assert is_ssh_key(b"# a comment\n" + key_format + b" " + blob) is True
    assert is_ssh_key(b"not a ssh key") is False


@pytest.mark.parametrize(
    "secret",
    (
        b"not a ssh key",
        b"ssh-rsa",
        b"ssh-rsa not-a-key",
        b"my ssh-rsa passphrase is strong",
        b"use ssh-ed25519 or ecdsa-sha2-nistp256 for auth",
    ),
)
def test_is_ssh_key_accepts_non_key_secrets(secret: bytes) -> None:
    assert is_ssh_key(secret) is False
