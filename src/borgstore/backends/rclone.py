"""
Borgstore backend for rclone
"""

import os
import re
import requests
import subprocess
import json
import secrets
from typing import Iterator
import threading

from ._base import BackendBase, ItemInfo, validate_name
from .errors import (
    BackendError,
    BackendDoesNotExist,
    BackendMustNotBeOpen,
    BackendMustBeOpen,
    BackendAlreadyExists,
    ObjectNotFound,
)
from ..constants import TMP_SUFFIX

# rclone binary - expected to be on the path
RCLONE = "rclone"

# Debug HTTP requests and responses
if False:
    import logging
    import http.client as http_client

    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


def get_rclone_backend(url):
    """get rclone URL
    rclone://remote:
    rclone://remote:path
    """
    # Check rclone is on the path
    try:
        info = json.loads(subprocess.check_output([RCLONE, "rc", "--loopback", "core/version"]))
    except Exception:
        raise BackendDoesNotExist("rclone binary not found on the path or not working properly")
    if info["decomposed"] < [1, 57, 0]:
        raise BackendDoesNotExist(f"rclone binary too old - need at least version v1.57.0 - found {info['version']}")
    rclone_regex = r"""
        rclone://
        (?P<path>(.*))
    """
    m = re.match(rclone_regex, url, re.VERBOSE)
    if m:
        return Rclone(path=m["path"])


class Rclone(BackendBase):
    """Borgstore backend for rclone

    This uses the rclone rc API to control an rclone rcd process.
    """

    HOST = "localhost"

    def __init__(self, path, *, do_fsync=False):
        if not path.endswith(":") and not path.endswith("/"):
            path += "/"
        self.fs = path
        self.process = None
        self.url = None
        self.user = "borg"
        self.password = secrets.token_urlsafe(32)

    def open(self):
        """
        Start using the rclone server
        """
        if self.process:
            raise BackendMustNotBeOpen()
        # Open rclone rcd listening on a random port with random auth
        args = [
            RCLONE,
            "rcd",
            "--rc-user",
            self.user,
            "--rc-addr",
            self.HOST + ":0",
            "--rc-serve",
            "--use-server-modtime",
        ]
        env = os.environ.copy()
        env["RCLONE_RC_PASS"] = self.password  # pass password by env var so it isn't in process list
        self.process = subprocess.Popen(
            args, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=env
        )
        # Read the log line with the port in it
        line = self.process.stderr.readline()
        m = re.search(rb"(http://.*/)", line)
        if not m:
            raise BackendDoesNotExist(f"rclone rcd did not return URL in log line: {line}")
        self.url = m.group(1).decode("utf-8")

        def discard():
            """discard log output on stderr so we don't block the process"""
            while True:
                line = self.process.stderr.readline()
                if not line:
                    break  # Process has finished

        thread = threading.Thread(target=discard)
        thread.start()

    def close(self):
        """
        Stop using the rclone server
        """
        if not self.process:
            raise BackendMustBeOpen()
        self.process.terminate()
        self.process = None
        self.url = None

    def _requests(self, fn, *args, **kwargs):
        """
        Runs a call to the requests function fn with *args and **kwargs

        It adds auth and decodes errors in a consistent way

        It returns the response object
        """
        if not self.process or not self.url:
            raise BackendMustBeOpen()
        r = fn(*args, auth=(self.user, self.password), **kwargs)
        if r.status_code == 404:
            raise ObjectNotFound(f"Not Found: error {r.status_code}: {r.text}")
        elif r.status_code not in (200, 206):
            raise BackendError(f"rclone rc command failed: error {r.status_code}: {r.text}")
        return r

    def _rpc(self, command, json_input, **kwargs):
        """
        Run the rclone command over the rclone API

        Additional kwargs may be passed to requests
        """
        if not self.url:
            raise BackendMustBeOpen()
        r = self._requests(requests.post, self.url + command, json=json_input, **kwargs)
        return r.json()

    def create(self):
        """create (initialize) the rclone storage"""
        if self.process:
            raise BackendMustNotBeOpen()
        with self:
            info = self.info("")
            if info.exists:
                raise BackendAlreadyExists(f"rclone storage base path already exists: {self.fs}")
            self.mkdir("")

    def destroy(self):
        """completely remove the rclone storage (and its contents)"""
        if self.process:
            raise BackendMustNotBeOpen()
        with self:
            info = self.info("")
            if not info.exists:
                raise BackendDoesNotExist(f"rclone storage base path does not exist: {self.fs}")
            self._rpc("operations/purge", {"fs": self.fs, "remote": ""})

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def mkdir(self, name: str) -> None:
        """create directory/namespace <name>"""
        validate_name(name)
        self._rpc("operations/mkdir", {"fs": self.fs, "remote": name})

    def rmdir(self, name: str) -> None:
        """remove directory/namespace <name>"""
        validate_name(name)
        self._rpc("operations/rmdir", {"fs": self.fs, "remote": name})

    def _to_item_info(self, remote, item):
        """Converts an rclone item at remote into a borgstore ItemInfo"""
        if item is None:
            return ItemInfo(name=os.path.basename(remote), exists=False, directory=False, size=0)
        name = item["Name"]
        size = item["Size"]
        directory = item["IsDir"]
        return ItemInfo(name=name, exists=True, size=size, directory=directory)

    def info(self, name) -> ItemInfo:
        """return information about <name>"""
        validate_name(name)
        try:
            result = self._rpc(
                "operations/stat",
                {"fs": self.fs, "remote": name, "opt": {"recurse": False, "noModTime": True, "noMimeType": True}},
            )
            item = result["item"]
        except ObjectNotFound:
            item = None
        return self._to_item_info(name, item)

    def load(self, name: str, *, size=None, offset=0) -> bytes:
        """load value from <name>"""
        validate_name(name)
        headers = {}
        if size is not None or offset > 0:
            if size is not None:
                headers["Range"] = f"bytes={offset}-{offset+size-1}"
            else:
                headers["Range"] = f"bytes={offset}-"
        r = self._requests(requests.get, f"{self.url}[{self.fs}]/{name}", headers=headers)
        return r.content

    def store(self, name: str, value: bytes) -> None:
        """store <value> into <name>"""
        validate_name(name)
        files = {"file": (os.path.basename(name), value, "application/octet-stream")}
        params = {"fs": self.fs, "remote": os.path.dirname(name)}
        self._rpc("operations/uploadfile", None, params=params, files=files)

    def delete(self, name: str) -> None:
        """delete <name>"""
        validate_name(name)
        self._rpc("operations/deletefile", {"fs": self.fs, "remote": name})

    def move(self, curr_name: str, new_name: str) -> None:
        """rename curr_name to new_name (overwrite target)"""
        validate_name(curr_name)
        validate_name(new_name)
        self._rpc(
            "operations/movefile", {"srcFs": self.fs, "srcRemote": curr_name, "dstFs": self.fs, "dstRemote": new_name}
        )

    def list(self, name: str) -> Iterator[ItemInfo]:
        """list the contents of <name>, non-recursively.

        Does not yield TMP_SUFFIX items - usually they are either not finished
        uploading or they are leftover crap from aborted uploads.

        The yielded ItemInfos are sorted alphabetically by name.
        """
        validate_name(name)
        result = self._rpc(
            "operations/list",
            {"fs": self.fs, "remote": name, "opt": {"recurse": False, "noModTime": True, "noMimeType": True}},
        )
        for item in result["list"]:
            name = item["Name"]
            if name.endswith(TMP_SUFFIX):
                continue
            yield self._to_item_info(name, item)