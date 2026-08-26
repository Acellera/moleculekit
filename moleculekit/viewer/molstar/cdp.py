"""Minimal Chrome DevTools Protocol client over a raw WebSocket.

Only the standard library is used. This exists so headless rendering needs no
browser-automation dependency: the sandbox images moleculekit runs in ship a
chromium binary but neither playwright nor node.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import time
import urllib.request

_TEXT, _CONTINUATION, _CLOSE, _PING, _PONG = 0x1, 0x0, 0x8, 0x9, 0xA
_READ_CHUNK = 1 << 20


def _encode_masked_frame(payload: bytes, opcode: int) -> bytes:
    """Encode a payload as a masked client-to-server WebSocket frame.

    Parameters
    ----------
    payload : bytes
        The frame body.
    opcode : int
        The frame opcode (e.g., _TEXT, _PONG).

    Returns
    -------
    frame : bytes
        The complete frame, including header and masking key.
    """
    mask = os.urandom(4)
    n = len(payload)
    header = bytes([0x80 | opcode])
    if n < 126:
        header += struct.pack("!B", 0x80 | n)
    elif n < (1 << 16):
        header += struct.pack("!BH", 0x80 | 126, n)
    else:
        header += struct.pack("!BQ", 0x80 | 127, n)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return header + mask + masked


def encode_frame(payload: bytes) -> bytes:
    """Encode a payload as a masked client-to-server WebSocket text frame.

    Parameters
    ----------
    payload : bytes
        The UTF-8 encoded message body.

    Returns
    -------
    frame : bytes
        The complete frame, including header and masking key.
    """
    return _encode_masked_frame(payload, _TEXT)


class _FrameReader:
    """Reassembles server frames into whole messages."""

    def __init__(self, sock, buffered: bytes = b""):
        self._sock = sock
        self._buf = buffered

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(_READ_CHUNK)
            if not chunk:
                raise ConnectionError("websocket closed by peer")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def next_message(self) -> str:
        payload = b""
        expecting_continuation = False
        while True:
            b0, b1 = struct.unpack("!BB", self._read(2))
            fin, opcode = b0 & 0x80, b0 & 0x0F
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read(8))[0]
            body = self._read(length)
            # Control frames may arrive between fragments (RFC 6455 §5.4)
            if opcode == _PING:
                self._sock.sendall(_encode_masked_frame(body, _PONG))
                continue
            if opcode == _PONG:
                continue
            if opcode == _CLOSE:
                raise ConnectionError("websocket closed by peer")
            # Data frames must follow the fragmentation protocol
            if expecting_continuation:
                if opcode != _CONTINUATION:
                    raise ConnectionError(f"expected continuation frame, got opcode {opcode}")
            payload += body
            if fin:
                return payload.decode("utf-8")
            expecting_continuation = True


class WS:
    """A CDP session over one WebSocket connection."""

    def __init__(self, url: str, timeout: float = 300.0):
        """Open a WebSocket connection to a devtools endpoint.

        Parameters
        ----------
        url : str
            The WebSocket URL (e.g., ws://localhost:9222/devtools/browser/...).
        timeout : float
            Socket timeout in seconds.

        Raises
        ------
        ConnectionError
            If the WebSocket handshake fails or devtools closes the connection.
        """
        _, _, rest = url.partition("://")
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self._sock = socket.create_connection((host, int(port or 80)))
        self._sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        self._sock.sendall(
            (
                f"GET /{path} HTTP/1.1\r\n"
                f"Host: {hostport}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii")
        )
        handshake = b""
        while b"\r\n\r\n" not in handshake:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("devtools closed the connection during handshake")
            handshake += chunk
        status, _, _ = handshake.partition(b"\r\n")
        if b"101" not in status:
            raise ConnectionError(f"websocket upgrade refused: {status!r}")
        self._reader = _FrameReader(self._sock, handshake.split(b"\r\n\r\n", 1)[1])
        self._next_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        """Send a CDP command and return its result, ignoring unrelated events.

        Parameters
        ----------
        method : str
            The CDP method name (e.g., 'Page.navigate').
        params : dict or None
            Method parameters.

        Returns
        -------
        result : dict
            The result object from the CDP response.

        Raises
        ------
        RuntimeError
            If the CDP endpoint returns an error.
        """
        self._next_id += 1
        message_id = self._next_id
        self._sock.sendall(
            encode_frame(
                json.dumps(
                    {"id": message_id, "method": method, "params": params or {}}
                ).encode("utf-8")
            )
        )
        while True:
            message = json.loads(self._reader.next_message())
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method} failed: {message['error']}")
            return message.get("result", {})

    def evaluate(self, expression: str, timeout_ms: int = 300000):
        """Evaluate JavaScript in the page and return its value.

        Promises are awaited. A JavaScript exception is re-raised here rather
        than returning a value, so a failed render cannot masquerade as a
        successful one.
        """
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
                "timeout": timeout_ms,
            },
        )
        if "exceptionDetails" in result:
            raise RuntimeError(f"page raised: {result['exceptionDetails']}")
        return result["result"].get("value")

    def set_timeout(self, timeout: float) -> None:
        """Change the socket's read timeout for subsequent calls.

        Parameters
        ----------
        timeout : float
            New socket timeout in seconds.
        """
        self._sock.settimeout(timeout)

    def close(self) -> None:
        """Close the WebSocket connection."""
        self._sock.close()


def page_target_url(port: int, timeout: float = 30.0) -> str:
    """Wait for devtools to come up and return the first page target's ws url.

    Parameters
    ----------
    port : int
        The port chromium was told to serve the devtools protocol on.
    timeout : float
        Seconds to keep polling before giving up.

    Returns
    -------
    url : str
        The websocket debugger url of the first page target.

    Raises
    ------
    TimeoutError
        If no page target appears within ``timeout``.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/list", timeout=2
            ) as response:
                for target in json.load(response):
                    if target.get("type") == "page" and target.get(
                        "webSocketDebuggerUrl"
                    ):
                        return target["webSocketDebuggerUrl"]
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(
        f"chromium devtools did not expose a page target on port {port} "
        f"within {timeout:g}s"
    )
