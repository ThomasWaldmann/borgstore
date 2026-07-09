"""
Tests for the SFTP backend's connection-loss recovery (see issue #167).

These tests do not need a real SFTP server: they exercise the with_reconnect decorator and the
_is_connection_lost helper in isolation, simulating a dead connection.
"""

import errno
import socket

import pytest

paramiko = pytest.importorskip("paramiko")

from borgstore.backends.errors import BackendError, ObjectNotFound
from borgstore.backends.sftp import Sftp, with_reconnect, _is_connection_lost


@pytest.mark.parametrize(
    "exc, expected",
    [
        (socket.timeout(), True),
        (EOFError(), True),
        (ConnectionResetError(), True),
        (paramiko.SSHException("Server connection dropped"), True),
        (OSError(errno.EPIPE, "broken pipe"), True),
        (OSError(errno.ECONNRESET, "reset"), True),
        # legit results, must NOT be treated as connection loss:
        (FileNotFoundError(errno.ENOENT, "no such file"), False),
        (PermissionError(errno.EACCES, "denied"), False),
        (ObjectNotFound("x"), False),
        (ValueError("nope"), False),
    ],
)
def test_is_connection_lost(exc, expected):
    assert _is_connection_lost(exc) is expected


class FakeSftp:
    """Minimal stand-in providing exactly what with_reconnect touches."""

    def __init__(self, reconnect_tries=3):
        self.opened = True
        self.reconnect_tries = reconnect_tries
        self.reconnect_wait = 0  # do not slow down the tests
        self.reconnects = 0
        self.reconnect_should_fail = False

    def _reconnect(self):
        self.reconnects += 1
        if self.reconnect_should_fail:
            raise socket.timeout()


def test_retry_succeeds_after_reconnect():
    obj = FakeSftp()
    calls = {"n": 0}

    @with_reconnect
    def op(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout()  # first call: connection is dead
        return "ok"  # after reconnect: works

    assert op(obj) == "ok"
    assert obj.reconnects == 1
    assert calls["n"] == 2


def test_no_retry_for_non_connection_error():
    obj = FakeSftp()

    @with_reconnect
    def op(self):
        raise ObjectNotFound("missing")

    with pytest.raises(ObjectNotFound):
        op(obj)
    assert obj.reconnects == 0  # ObjectNotFound must not trigger a reconnect


def test_gives_up_after_all_tries():
    obj = FakeSftp(reconnect_tries=3)

    @with_reconnect
    def op(self):
        raise socket.timeout()  # never recovers

    with pytest.raises(BackendError):
        op(obj)
    assert obj.reconnects == 3


def test_swallow_not_found_on_retry():
    """delete/move: ObjectNotFound after a reconnect means the earlier attempt already did it."""
    obj = FakeSftp()
    calls = {"n": 0}

    @with_reconnect(swallow_not_found=True)
    def op(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout()  # connection died (the op may already have succeeded)
        raise ObjectNotFound("gone")  # retry: object is already gone -> treat as success

    assert op(obj) is None
    assert obj.reconnects == 1


def test_object_not_found_propagates_on_first_attempt_even_when_swallowing():
    """A genuine ObjectNotFound (no connection loss) must still propagate."""
    obj = FakeSftp()

    @with_reconnect(swallow_not_found=True)
    def op(self):
        raise ObjectNotFound("gone")

    with pytest.raises(ObjectNotFound):
        op(obj)
    assert obj.reconnects == 0


def test_object_not_found_on_retry_propagates_without_swallow():
    """Without swallow_not_found, ObjectNotFound on the retry path propagates as-is."""
    obj = FakeSftp()
    calls = {"n": 0}

    @with_reconnect
    def op(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout()
        raise ObjectNotFound("gone")

    with pytest.raises(ObjectNotFound):
        op(obj)
    assert obj.reconnects == 1


def test_not_opened_does_not_reconnect():
    obj = FakeSftp()
    obj.opened = False

    @with_reconnect
    def op(self):
        raise socket.timeout()

    with pytest.raises(socket.timeout):
        op(obj)
    assert obj.reconnects == 0


def test_reconnect_restores_working_directory(monkeypatch):
    """After a reconnect we must chdir back into base_path so relative names keep working."""
    be = Sftp(hostname="example", path="some/base/path")

    class FakeClient:
        def __init__(self):
            self.cwd = None

        def chdir(self, path):
            self.cwd = path

    connects = {"n": 0}

    def fake_connect():
        connects["n"] += 1
        be.client = FakeClient()

    def fake_disconnect():
        be.client = None

    monkeypatch.setattr(be, "_connect", fake_connect)
    monkeypatch.setattr(be, "_disconnect", fake_disconnect)

    be._reconnect()

    assert connects["n"] == 1
    assert be.client.cwd == "some/base/path"
