import json
import socket
import time
import urllib.request
from urllib.error import HTTPError

import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar import server as molstar_server
from moleculekit.viewer.molstar.scene import build_scene


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
    """Read one SSE 'data:' line from `resp` and return the parsed JSON.

    Leftover bytes past the end of the event returned are kept on `resp`
    (a `_sse_buf` attribute) instead of being dropped when this function
    returns, so a caller that calls this repeatedly on the same `resp`
    (replaying several events in a row, for instance) still sees an event
    whose bytes happened to arrive in the same read as the one just
    consumed, rather than starting the next call's own read from a buffer
    that silently lost bytes.

    (An earlier version of this helper tried to pre-check readability with
    ``select()`` before every read, to avoid blocking past `timeout`. That
    does not work: `resp.fp` is a buffered reader, so bytes already sitting
    in its userspace buffer, e.g. read ahead alongside the response headers
    during `urlopen()` itself, are invisible to ``select()``, which only
    sees the OS socket. ``select()`` can then report "not readable" and
    time out even though `resp.read1()` would have returned instantly.
    `read1()` already does the right thing on its own: it returns whatever
    is buffered without blocking, and only touches the socket when the
    buffer is empty. So the wait here is bounded by `resp`'s own
    socket-level `timeout=` (see `urllib.request.urlopen`), not by this
    function's own bookkeeping; a caller that wants a specific bound should
    open `resp` with a comparable or larger `timeout=`.)
    """
    deadline = time.time() + timeout
    buf = getattr(resp, "_sse_buf", b"")
    try:
        while True:
            while b"\n\n" in buf:
                block, _, buf = buf.partition(b"\n\n")
                for line in block.split(b"\n"):
                    if line.startswith(b"data:"):
                        return json.loads(line[5:].strip().decode("utf-8"))
                    # heartbeat (": heartbeat"), skip and keep reading
            if time.time() >= deadline:
                raise AssertionError("No SSE event received within timeout")
            try:
                chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
            except TimeoutError:
                # A single blocking read that times out leaves the
                # underlying socket unusable for any further read (CPython
                # marks it and any later call raises OSError instead), so
                # this cannot retry on the same `resp`: fail cleanly here.
                raise AssertionError("No SSE event received within timeout") from None
            if not chunk:
                time.sleep(0.05)
                continue
            buf += chunk
    finally:
        resp._sse_buf = buf


def _wait_until(condition, timeout=5.0, interval=0.02):
    """Poll `condition` (a zero-arg callable) until it is truthy.

    Returns True as soon as `condition()` is truthy, False once `timeout`
    seconds have elapsed without that happening. Used instead of a fixed
    `time.sleep()` to wait for a background thread's effect: the wait ends
    as soon as the effect is observed, not after a guessed duration that can
    be too short under load and is always longer than necessary otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def test_server_serves_index_html(fresh_server):
    url = f"http://127.0.0.1:{fresh_server.port}/"
    with urllib.request.urlopen(url, timeout=2) as resp:
        body = resp.read()
        assert resp.status == 200
        assert b"<html" in body.lower() or b"<!doctype html" in body.lower()


def test_session_mismatch_returns_410(fresh_server):
    url = f"http://127.0.0.1:{fresh_server.port}/events?session=WRONG"
    with pytest.raises(HTTPError) as exc:
        urllib.request.urlopen(url, timeout=2)
    assert exc.value.code == 410


def test_register_emits_topology_event(fresh_server):
    url = f"http://127.0.0.1:{fresh_server.port}/events?session={fresh_server.session}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    # timeout= bounds each individual blocking read _read_one_sse_event
    # makes below (see its docstring); keep it >= that call's own timeout.
    resp = urllib.request.urlopen(req, timeout=6)
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
    url = f"http://127.0.0.1:{fresh_server.port}/coords/{uid}/{slot.topo_hash}"
    with urllib.request.urlopen(url, timeout=2) as resp:
        blob = resp.read()
        assert resp.headers["Content-Type"] == "application/octet-stream"
        arr = np.frombuffer(blob, dtype="<f4")
        assert arr.size == 2 * 3 * 1
        assert arr[0] == pytest.approx(7.0)


def test_coords_endpoint_stale_topohash_returns_404(fresh_server):
    mol = _make_mol()
    uid = molstar_server.register(mol)
    url = f"http://127.0.0.1:{fresh_server.port}/coords/{uid}/deadbeef"
    with pytest.raises(HTTPError) as exc:
        urllib.request.urlopen(url, timeout=2)
    assert exc.value.code == 404


def test_unregister_endpoint_removes_slot(fresh_server):
    mol = _make_mol()
    uid = molstar_server.register(mol)
    assert uid in molstar_server.get_registry().slots
    url = f"http://127.0.0.1:{fresh_server.port}/unregister/{uid}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=2) as resp:
        assert resp.status == 204
    assert uid not in molstar_server.get_registry().slots


def test_port_walkup(monkeypatch):
    molstar_server.shutdown_for_tests()
    # Occupy a pristine, OS-assigned port and point the server's range at it.
    # Binding port 0 always lands on a fresh port with no leftover socket state,
    # so this never trips over TIME_WAIT/CLOSE_WAIT lingering from earlier tests
    # (which made rebinding a hardcoded port fail on macOS).
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.bind(("127.0.0.1", 0))
    taken = occupier.getsockname()[1]
    occupier.listen(1)
    monkeypatch.setattr(molstar_server, "_PORT_RANGE", range(taken, taken + 11))
    try:
        state = molstar_server.start_for_tests(open_browser=False)
        assert state.port != taken
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

    url = f"http://127.0.0.1:{state.port}/events?session={state.session}"
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


def test_topology_event_carries_the_scene_description():
    """The viewer cannot honour mol.reps unless the scene reaches it."""
    from moleculekit.viewer.molstar.server import _topology_event
    from moleculekit.viewer.molstar.registry import Registry

    mol = Molecule().empty(2)
    mol.element[:] = ["C", "O"]
    mol.name[:] = ["C1", "O2"]
    mol.resname[:] = "LIG"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.segid[:] = "L"
    mol.record[:] = "HETATM"
    mol.serial[:] = [1, 2]
    mol.coords = np.zeros((2, 3, 1), dtype=np.float32)
    mol.reps.add("name C1", style="VDW", color="red")

    registry = Registry()
    uid = registry.register(mol)
    event = _topology_event(registry.slots[uid])

    assert "scene" in event
    assert event["scene"]["components"][0]["representation"]["type"] == "spacefill"
    assert event["scene"]["components"][0]["color"] == {"uniform": "red"}


def test_bad_rep_selection_does_not_break_new_sse_connections(fresh_server):
    """One slot's build_scene() failure must not break a fresh /events replay.

    _serve_events replays a topology event per registered slot when a client
    connects. A rep selection that matches no atoms makes build_scene() raise
    for that one slot; that must not stop the other, valid slot's event from
    reaching the new connection.
    """
    mol_good = _make_mol()
    mol_bad = _make_mol()
    uid_good = molstar_server.register(mol_good)
    molstar_server.register(mol_bad)
    mol_bad.reps.add("name ZZZ_NOT_REAL", style="VDW")

    url = f"http://127.0.0.1:{fresh_server.port}/events?session={fresh_server.session}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    # timeout= bounds each individual blocking read _read_one_sse_event
    # makes below (see its docstring); keep it >= that call's own timeout.
    resp = urllib.request.urlopen(req, timeout=6)
    try:
        ev = _read_one_sse_event(resp, timeout=3.0)
        assert ev["type"] == "topology"
        assert ev["slot"] == uid_good
    finally:
        resp.close()


def test_bad_rep_selection_does_not_stop_other_slots_broadcasting(fresh_server):
    """One slot's build_scene() failure must not kill the monitor thread.

    A rep selection that matches no atoms, combined with a genuine topology
    change (formalcharge is a topo field), makes build_scene() raise inside
    the monitor loop's per-slot event construction. That must not stop the
    monitor thread from broadcasting later, unrelated changes on the other,
    valid slot.
    """
    mol_good = _make_mol()
    mol_bad = _make_mol()
    uid_good = molstar_server.register(mol_good)
    uid_bad = molstar_server.register(mol_bad)

    url = f"http://127.0.0.1:{fresh_server.port}/events?session={fresh_server.session}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    # timeout= bounds each individual blocking read _read_one_sse_event
    # makes below (see its docstring); keep it >= that call's own timeout
    # (the largest used on this connection is the 5.0 below).
    resp = urllib.request.urlopen(req, timeout=6)
    try:
        replayed = {
            _read_one_sse_event(resp, timeout=3.0)["slot"] for _ in range(2)
        }
        assert replayed == {uid_good, uid_bad}

        registry = molstar_server.get_registry()
        bad_topo_hash = registry.slots[uid_bad].topo_hash

        mol_bad.reps.add("name ZZZ_NOT_REAL", style="VDW")
        mol_bad.formalcharge[:] = [1, 0]
        # Wait for the monitor loop to actually poll and hit the bad slot,
        # rather than sleeping a guessed number of poll intervals: the
        # monitor's diff_and_snapshot() updates a changed slot's topo_hash
        # unconditionally, before the (possibly failing) _topology_event()
        # build for that slot is even attempted, so once this hash moves,
        # that tick has already reached and recovered from mol_bad's
        # build_scene() failure (both happen synchronously in the same
        # monitor-thread iteration, so there is no further race to win here).
        assert _wait_until(
            lambda: registry.slots[uid_bad].topo_hash != bad_topo_hash
        ), "monitor loop never processed the bad slot's topology change"

        mol_good.coords[0, 0, 0] = 42.0
        ev = _read_one_sse_event(resp, timeout=5.0)
        assert ev["type"] == "coords"
        assert ev["slot"] == uid_good
    finally:
        resp.close()


def test_view_sel_does_not_leak_into_later_scenes(fresh_server, monkeypatch):
    """A `sel=`/`style=`/`color=` view() adds a one-off representation to
    mol._tempreps (see molecule.py's view()). The VMD and NGL backends
    consume it and then call `self._tempreps.remove()`, making it one-shot;
    before this fix the molstar path never cleared it, so the server (which
    keeps a live reference to mol and rebuilds the scene from it on every
    later topology change) kept reusing that one-off selection forever: a
    single `mol.view(viewer="molstar", sel="protein")` silently made every
    later `view()`/`render()` in the session protein-only.

    Two `hold=True` calls exercise the reviewer's exact repro (they collect
    without rendering); the final non-hold call must both still use them for
    that one render and clear them immediately afterward.
    """
    captured = []
    real_broadcast = molstar_server._broadcast

    def _spy(state, event):
        captured.append(event)
        real_broadcast(state, event)

    monkeypatch.setattr(molstar_server, "_broadcast", _spy)

    mol = _make_mol()
    mol.view(viewer="molstar", sel="name C1", hold=True)
    mol.view(viewer="molstar", sel="name O2", style="Licorice", hold=True)
    assert len(mol._tempreps.replist) == 2  # collected while held, not yet rendered

    mol.view(viewer="molstar")  # hold=False: renders once, then must clear them

    # The render that just happened is the one and only broadcast so far,
    # and it must still reflect both temp reps: this is the timing check
    # that clearing does not erase them before they are used.
    assert len(captured) == 1
    assert len(captured[0]["scene"]["components"]) == 2

    assert mol._tempreps.replist == [], "sel=/style= temp reps leaked past view()"

    # A later, unrelated re-registration (standing in for the monitor
    # thread's topology-change broadcast) must fall back to mol.reps alone,
    # not silently re-include the one-off selection consumed above.
    molstar_server.register(mol)
    assert len(captured) == 2
    assert captured[1]["scene"] == build_scene(mol, background_color="white")
