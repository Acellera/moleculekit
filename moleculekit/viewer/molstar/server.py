"""Singleton HTTP server with SSE for the molstar viewer."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import secrets
import socket
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from moleculekit.viewer.molstar.registry import Registry, coords_to_bytes
from moleculekit.viewer.molstar.serialize import molecule_to_dict

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_PORT_RANGE = range(8765, 8776)
_POLL_INTERVAL = 0.5
_SHUTDOWN_WAIT = 5.0  # max seconds to wait for threads/port during shutdown
_WAKE = object()  # sentinel pushed to SSE queues to unblock them on shutdown


@dataclass
class ServerState:
    port: int
    session: str
    httpd: ThreadingHTTPServer
    server_thread: threading.Thread
    monitor_thread: threading.Thread
    registry: Registry
    subscribers: list[queue.Queue]
    subscribers_lock: threading.Lock
    stop_event: threading.Event
    handler_threads: set[threading.Thread]


_state: ServerState | None = None
_state_lock = threading.Lock()


def get_or_start_server(open_browser: bool = True) -> ServerState:
    """Start the singleton server on first call; subsequent calls are no-ops."""
    global _state
    with _state_lock:
        if _state is None:
            _state = _start(open_browser=open_browser)
            atexit.register(shutdown_for_tests)
        return _state


def get_registry() -> Registry:
    return get_or_start_server().registry


def register(mol) -> str:
    """Register a Molecule with the running server; returns its slot uuid."""
    state = get_or_start_server()
    uid = state.registry.register(mol)
    slot = state.registry.slots[uid]
    _broadcast(state, _topology_event(slot))
    return uid


def _start(open_browser: bool) -> ServerState:
    port = _bind_first_free_port()
    session = secrets.token_urlsafe(12)
    registry = Registry()
    subscribers: list[queue.Queue] = []
    subscribers_lock = threading.Lock()
    stop_event = threading.Event()
    handler_threads: set[threading.Thread] = set()

    handler_factory = _make_handler_class(
        session=session,
        registry=registry,
        subscribers=subscribers,
        subscribers_lock=subscribers_lock,
        stop_event=stop_event,
        handler_threads=handler_threads,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_factory)

    server_thread = threading.Thread(
        target=httpd.serve_forever, name="molstar-http", daemon=True
    )
    server_thread.start()

    monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(registry, subscribers, subscribers_lock, stop_event),
        name="molstar-monitor",
        daemon=True,
    )
    monitor_thread.start()

    state = ServerState(
        port=port, session=session, httpd=httpd,
        server_thread=server_thread, monitor_thread=monitor_thread,
        registry=registry, subscribers=subscribers,
        subscribers_lock=subscribers_lock, stop_event=stop_event,
        handler_threads=handler_threads,
    )

    if open_browser:
        webbrowser.open_new_tab(f"http://localhost:{port}/?session={session}")
    return state


def _bind_first_free_port() -> int:
    for port in _PORT_RANGE:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Match ThreadingHTTPServer's allow_reuse_address so the probe accepts
        # the same ports the server can bind, instead of walking past (and
        # exhausting) ports merely lingering in TIME_WAIT.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            s.close()
            continue
        s.close()
        return port
    raise RuntimeError(
        f"No free port in {_PORT_RANGE.start}..{_PORT_RANGE.stop - 1} "
        f"for the molstar viewer."
    )


def _monitor_loop(
    registry: Registry,
    subscribers: list[queue.Queue],
    subscribers_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(_POLL_INTERVAL):
        try:
            changes = registry.diff_and_snapshot()
        except Exception:
            logger.exception("molstar monitor diff failed")
            continue
        for kind, uid, _hints in changes:
            slot = registry.slots.get(uid)
            if slot is None:
                continue
            if kind == "topology":
                ev = _topology_event(slot)
            else:
                ev = _coords_event(slot)
            _broadcast_to(subscribers, subscribers_lock, ev)


def _topology_event(slot) -> dict:
    return {
        "type": "topology",
        "slot": slot.uuid,
        "label": slot.label,
        "mol": molecule_to_dict(slot.mol_ref),
        "coords_url": f"/coords/{slot.uuid}/{slot.topo_hash}",
        "numFrames": int(slot.mol_ref.coords.shape[2]),
    }


def _coords_event(slot) -> dict:
    return {
        "type": "coords",
        "slot": slot.uuid,
        "coords_url": f"/coords/{slot.uuid}/{slot.topo_hash}",
        "numFrames": int(slot.mol_ref.coords.shape[2]),
    }


def _broadcast(state: ServerState, event: dict) -> None:
    _broadcast_to(state.subscribers, state.subscribers_lock, event)


def _broadcast_to(subscribers, subscribers_lock, event: dict) -> None:
    payload = json.dumps(event)
    with subscribers_lock:
        for q in list(subscribers):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def _make_handler_class(
    *, session, registry, subscribers, subscribers_lock, stop_event, handler_threads,
):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug("%s - %s", self.address_string(), format % args)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path.startswith("/assets/"):
                self._serve_static(path.lstrip("/"))
            elif path == "/events":
                self._serve_events(parsed.query)
            elif path.startswith("/coords/"):
                self._serve_coords(path)
            else:
                self.send_error(404, "Not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/unregister/"):
                uid = parsed.path[len("/unregister/"):]
                registry.remove(uid)
                _broadcast_to(
                    subscribers, subscribers_lock,
                    {"type": "remove", "slot": uid},
                )
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404, "Not found")

        def _serve_static(self, rel: str):
            target = (_STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(_STATIC_DIR.resolve())):
                self.send_error(403, "Forbidden")
                return
            if not target.is_file():
                self.send_error(404, "Not found")
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _guess_mime(target.suffix))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self, query: str):
            qs = parse_qs(query)
            client_session = (qs.get("session") or [""])[0]
            if client_session != session:
                self.send_error(410, "Session mismatch - refresh page")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q: queue.Queue = queue.Queue(maxsize=128)
            with subscribers_lock:
                subscribers.append(q)
                handler_threads.add(threading.current_thread())
            for slot in list(registry.slots.values()):
                try:
                    q.put_nowait(json.dumps(_topology_event(slot)))
                except queue.Full:
                    pass
            try:
                while not stop_event.is_set():
                    try:
                        payload = q.get(timeout=10)
                    except queue.Empty:
                        try:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        continue
                    if payload is _WAKE:
                        # Shutdown nudge: the while-condition ends the stream.
                        continue
                    try:
                        self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)
                    handler_threads.discard(threading.current_thread())

        def _serve_coords(self, path: str):
            parts = path.split("/")
            if len(parts) != 4:
                self.send_error(404, "Not found")
                return
            _, _, uid, topohash = parts
            slot = registry.slots.get(uid)
            if slot is None or slot.topo_hash != topohash:
                self.send_error(404, "Not found")
                return
            blob = coords_to_bytes(slot.mol_ref)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    return Handler


def _guess_mime(suffix: str) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".js":   "application/javascript",
        ".mjs":  "application/javascript",
        ".css":  "text/css",
        ".svg":  "image/svg+xml",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".json": "application/json",
    }.get(suffix.lower(), "application/octet-stream")


# --- test helpers ---------------------------------------------------------

def start_for_tests(open_browser: bool = False) -> ServerState:
    global _state
    with _state_lock:
        if _state is not None:
            _stop(_state)
            _state = None
        _state = _start(open_browser=open_browser)
        return _state


def shutdown_for_tests() -> None:
    global _state
    with _state_lock:
        if _state is not None:
            _stop(_state)
            _state = None


def _stop(state: ServerState) -> None:
    state.stop_event.set()
    # Unblock SSE handler loops parked in q.get() so they observe stop_event and
    # return now. Otherwise they linger until the queue timeout, keeping their
    # connection (and the port) open, which makes the next bind fail on macOS.
    with state.subscribers_lock:
        subscribers = list(state.subscribers)
        handler_threads = list(state.handler_threads)
    for q in subscribers:
        try:
            q.put_nowait(_WAKE)
        except queue.Full:
            pass
    try:
        state.httpd.shutdown()
        state.httpd.server_close()
    except Exception:
        pass
    state.server_thread.join(timeout=_SHUTDOWN_WAIT)
    state.monitor_thread.join(timeout=_SHUTDOWN_WAIT)
    # Wait for the woken SSE handlers to return: socketserver only closes their
    # sockets (and frees the port) once their thread exits.
    for t in handler_threads:
        t.join(timeout=_SHUTDOWN_WAIT)
