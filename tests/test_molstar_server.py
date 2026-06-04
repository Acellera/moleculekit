import json
import socket
import time
import urllib.request
from urllib.error import HTTPError

import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar import server as molstar_server


@pytest.fixture
def fresh_server():
    molstar_server.shutdown_for_tests()
    state = molstar_server.start_for_tests(open_browser=False)
    yield state
    molstar_server.shutdown_for_tests()


def _make_mol():
    mol = Molecule().empty(2)
    mol.element[:] = ["C", "O"]
    mol.name[:] = ["C1", "O2"]
    mol.resname[:] = ["LIG", "LIG"]
    mol.resid[:] = [1, 1]
    mol.chain[:] = ["A", "A"]
    mol.segid[:] = ["L", "L"]
    mol.record[:] = ["HETATM", "HETATM"]
    mol.serial[:] = [1, 2]
    mol.bonds = np.array([[0, 1]], dtype=np.uint32)
    mol.bondtype = np.array(["1"], dtype=object)
    mol.coords = np.zeros((2, 3, 1), dtype=np.float32)
    return mol


def _read_one_sse_event(resp, timeout=3.0):
    """Read one SSE 'data:' line from `resp` and return the parsed JSON."""
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
        if not chunk:
            time.sleep(0.05)
            continue
        buf += chunk
        while b"\n\n" in buf:
            block, _, buf = buf.partition(b"\n\n")
            for line in block.split(b"\n"):
                if line.startswith(b"data:"):
                    return json.loads(line[5:].strip().decode("utf-8"))
                # heartbeat (": heartbeat") — skip and keep reading
    raise AssertionError("No SSE event received within timeout")


def test_server_serves_index_html(fresh_server):
    url = f"http://localhost:{fresh_server.port}/"
    with urllib.request.urlopen(url, timeout=2) as resp:
        body = resp.read()
        assert resp.status == 200
        assert b"<html" in body.lower() or b"<!doctype html" in body.lower()


def test_session_mismatch_returns_410(fresh_server):
    url = f"http://localhost:{fresh_server.port}/events?session=WRONG"
    with pytest.raises(HTTPError) as exc:
        urllib.request.urlopen(url, timeout=2)
    assert exc.value.code == 410


def test_register_emits_topology_event(fresh_server):
    url = f"http://localhost:{fresh_server.port}/events?session={fresh_server.session}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    resp = urllib.request.urlopen(req, timeout=2)
    try:
        mol = _make_mol()
        molstar_server.register(mol)
        ev = _read_one_sse_event(resp, timeout=3.0)
        assert ev["type"] == "topology"
        assert ev["mol"]["numAtoms"] == 2
        assert ev["coords_url"].startswith("/coords/")
    finally:
        resp.close()


def test_coords_endpoint_returns_float32_blob(fresh_server):
    mol = _make_mol()
    mol.coords[0, 0, 0] = 7.0
    uid = molstar_server.register(mol)
    slot = molstar_server.get_registry().slots[uid]
    url = f"http://localhost:{fresh_server.port}/coords/{uid}/{slot.topo_hash}"
    with urllib.request.urlopen(url, timeout=2) as resp:
        blob = resp.read()
        assert resp.headers["Content-Type"] == "application/octet-stream"
        arr = np.frombuffer(blob, dtype="<f4")
        assert arr.size == 2 * 3 * 1
        assert arr[0] == pytest.approx(7.0)


def test_coords_endpoint_stale_topohash_returns_404(fresh_server):
    mol = _make_mol()
    uid = molstar_server.register(mol)
    url = f"http://localhost:{fresh_server.port}/coords/{uid}/deadbeef"
    with pytest.raises(HTTPError) as exc:
        urllib.request.urlopen(url, timeout=2)
    assert exc.value.code == 404


def test_unregister_endpoint_removes_slot(fresh_server):
    mol = _make_mol()
    uid = molstar_server.register(mol)
    assert uid in molstar_server.get_registry().slots
    url = f"http://localhost:{fresh_server.port}/unregister/{uid}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=2) as resp:
        assert resp.status == 204
    assert uid not in molstar_server.get_registry().slots


def test_port_walkup():
    molstar_server.shutdown_for_tests()
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupier.bind(("127.0.0.1", 8765))
    occupier.listen(1)
    try:
        state = molstar_server.start_for_tests(open_browser=False)
        assert state.port != 8765
    finally:
        occupier.close()
        molstar_server.shutdown_for_tests()


def test_shutdown_stops_open_sse_handler():
    """Shutdown must terminate an open SSE handler instead of leaving it parked.

    The handler used to block in ``q.get(timeout=...)`` and only re-check the
    stop flag afterwards, so its connection (and the bound port) lingered for
    the full timeout. On macOS the next ``bind`` then failed with EADDRINUSE
    because SO_REUSEADDR cannot reuse a still-open socket.
    """
    molstar_server.shutdown_for_tests()
    state = molstar_server.start_for_tests(open_browser=False)

    url = f"http://localhost:{state.port}/events?session={state.session}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    resp = urllib.request.urlopen(req, timeout=2)
    try:
        time.sleep(0.3)  # let the handler thread park in q.get()
        handlers = list(state.handler_threads)
        assert handlers and all(t.is_alive() for t in handlers)
    finally:
        resp.close()

    molstar_server.shutdown_for_tests()

    # Shutdown must have joined the SSE handler, not left it parked for ~10s.
    assert all(not t.is_alive() for t in handlers)
