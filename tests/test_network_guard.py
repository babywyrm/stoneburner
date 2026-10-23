"""The suite's network guard: tests stay on loopback test doubles."""

from __future__ import annotations

import socket

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("address", "recorded"),
    [
        (("192.0.2.1", 443), "192.0.2.1:443"),
        (("127.0.0.1", 11434), "127.0.0.1:11434"),
    ],
)
def test_the_guard_refuses_and_records(_no_network, address, recorded):
    with socket.socket() as sock, pytest.raises(ConnectionRefusedError):
        sock.settimeout(1)
        sock.connect(address)
    assert _no_network == [recorded]
    _no_network.clear()


def test_a_swallowed_refusal_still_fails_the_test(pytester):
    pytester.makeconftest("from tests.conftest import _no_network  # noqa: F401")
    pytester.makepyfile(
        """
        import socket

        def test_health_check_style():
            try:
                socket.create_connection(("192.0.2.1", 443), timeout=1)
            except OSError:
                pass
        """
    )
    result = pytester.runpytest_inprocess(
        "-p", "no:cacheprovider", "-o", "asyncio_default_fixture_loop_scope=function"
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*test reached the network: 192.0.2.1:443*"])


def test_the_guard_allows_an_ephemeral_loopback_port(_no_network):
    with socket.create_server(("127.0.0.1", 0)) as server:
        with socket.create_connection(server.getsockname()):
            pass
    assert _no_network == []
