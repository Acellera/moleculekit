import json
import struct

import pytest

from moleculekit.viewer.molstar import cdp


def _server_frame(payload: bytes, opcode: int = 0x1, fin: bool = True) -> bytes:
    """Build a server-to-client frame (unmasked, as the protocol requires)."""
    header = bytes([(0x80 if fin else 0) | opcode])
    n = len(payload)
    if n < 126:
        header += struct.pack("!B", n)
    elif n < (1 << 16):
        header += struct.pack("!BH", 126, n)
    else:
        header += struct.pack("!BQ", 127, n)
    return header + payload


class FakeSocket:
    """Minimal stand-in for a connected socket."""

    def __init__(self, inbound: bytes = b""):
        self.inbound = inbound
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        chunk, self.inbound = self.inbound[:n], self.inbound[n:]
        return chunk

    def settimeout(self, _):
        pass

    def close(self):
        pass


def test_encode_frame_masks_the_payload():
    """Client frames must be masked, and unmasking must recover the payload."""
    frame = cdp.encode_frame(b"hello")

    assert frame[0] == 0x81  # FIN + text opcode
    assert frame[1] & 0x80  # mask bit set
    assert frame[1] & 0x7F == 5
    mask, body = frame[2:6], frame[6:]
    assert bytes(b ^ mask[i % 4] for i, b in enumerate(body)) == b"hello"


def test_encode_frame_uses_extended_length_for_large_payloads():
    frame = cdp.encode_frame(b"x" * 70000)
    assert frame[1] & 0x7F == 127
    assert struct.unpack("!Q", frame[2:10])[0] == 70000


def test_reader_reassembles_fragmented_frames():
    """A message split across continuation frames arrives whole."""
    sock = FakeSocket(
        _server_frame(b'{"id":1,"resu', opcode=0x1, fin=False)
        + _server_frame(b'lt":{"ok":true}}', opcode=0x0, fin=True)
    )
    reader = cdp._FrameReader(sock)

    assert json.loads(reader.next_message()) == {"id": 1, "result": {"ok": True}}


def test_reader_answers_ping_with_pong_and_keeps_reading():
    sock = FakeSocket(
        _server_frame(b"", opcode=0x9) + _server_frame(b'{"id":2}', opcode=0x1)
    )
    reader = cdp._FrameReader(sock)

    assert json.loads(reader.next_message()) == {"id": 2}
    assert sock.sent[0] == 0x8A  # a pong went back


def test_reader_pong_echoes_ping_payload():
    """Pong frames must echo the ping's payload (RFC 6455 §5.5.3)."""
    ping_payload = b"keepalive-nonce-1234"
    sock = FakeSocket(
        _server_frame(ping_payload, opcode=0x9)
        + _server_frame(b'{"id":3}', opcode=0x1)
    )
    reader = cdp._FrameReader(sock)

    assert json.loads(reader.next_message()) == {"id": 3}

    # Verify the pong echoed the ping's payload, with correct frame format.
    pong = sock.sent
    assert pong[0] == 0x8A  # FIN + pong opcode
    assert pong[1] & 0x80  # mask bit set
    pong_len = pong[1] & 0x7F
    assert pong_len == len(ping_payload)
    mask = pong[2:6]
    pong_body = pong[6:]
    unmasked = bytes(b ^ mask[i % 4] for i, b in enumerate(pong_body))
    assert unmasked == ping_payload


def test_reader_handles_ping_interleaved_in_fragmented_message():
    """Control frames may arrive between fragments (RFC 6455 §5.4)."""
    ping_payload = b"keepalive-nonce-5678"
    sock = FakeSocket(
        _server_frame(b'{"id":4,"res', opcode=0x1, fin=False)
        + _server_frame(ping_payload, opcode=0x9, fin=True)
        + _server_frame(b'ult":"ok"}', opcode=0x0, fin=True)
    )
    reader = cdp._FrameReader(sock)

    assert json.loads(reader.next_message()) == {"id": 4, "result": "ok"}

    # Verify the pong echoed the ping's payload.
    pong = sock.sent
    assert pong[0] == 0x8A  # FIN + pong opcode
    assert pong[1] & 0x80  # mask bit set
    pong_len = pong[1] & 0x7F
    assert pong_len == len(ping_payload)
    mask = pong[2:6]
    pong_body = pong[6:]
    unmasked = bytes(b ^ mask[i % 4] for i, b in enumerate(pong_body))
    assert unmasked == ping_payload


def test_reader_raises_when_the_server_closes():
    sock = FakeSocket(_server_frame(b"", opcode=0x8))
    reader = cdp._FrameReader(sock)

    with pytest.raises(ConnectionError, match="closed"):
        reader.next_message()
