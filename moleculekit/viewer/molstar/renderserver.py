"""A render server: draw images for machines that cannot draw their own.

Rendering needs a browser, and it is far faster with a GPU. Neither is
available in a sandbox or on a cluster node, so this serves the drawing half
of ``render()`` over HTTP: a client turns its Molecule into a structure and a
scene description, both plain data, and posts them here. The browser stays on
this machine, which is what makes the arrangement worth having, since a GPU
render is roughly thirty times faster than a software one on a small container.

Run it on the machine with the GPU::

    python -m moleculekit.viewer.molstar.renderserver --port 8080

and point clients at it, either per call with ``render(..., server=...)`` or
for a whole environment::

    export MOLECULEKIT_RENDER_SERVER=http://gpuhost:8080

The service renders whatever it is sent and has no authentication, so bind it
to a private interface, which is what the default host does. Do not put it on a
public address.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from moleculekit.viewer.molstar.render import (
    QUALITY_PRESETS,
    _get_or_start,
    render_png,
)

logger = logging.getLogger(__name__)

#: Refuse bodies larger than this. A structure arrives base64 encoded, so this
#: allows a very large complex while still bounding what one request can cost.
MAX_REQUEST_BYTES = 64 * 1024 * 1024

#: One browser serves every request, and it is a singleton with a single
#: devtools connection, so renders are taken one at a time.
_render_lock = threading.Lock()


def _validate(body: dict) -> dict:
    """Check a request body and return the arguments to render with.

    Parameters
    ----------
    body : dict
        The decoded request body.

    Returns
    -------
    arguments : dict
        Keyword arguments for :func:`moleculekit.viewer.molstar.render.render_png`.

    Raises
    ------
    ValueError
        If a field is missing, of the wrong type, or out of range.
    """
    for field in ("width", "height"):
        if field not in body:
            raise ValueError(f"missing field {field!r}")
    if "objects" in body:
        # Several objects drawn together, each with its own scene.
        if not isinstance(body["objects"], list) or not body["objects"]:
            raise ValueError("objects must be a non-empty list")
        for obj in body["objects"]:
            if not isinstance(obj.get("structure"), str):
                raise ValueError("each object needs a base64 structure")
            if not isinstance(obj.get("scene"), dict):
                raise ValueError("each object needs a scene object")
    else:
        for field in ("structure", "scene"):
            if field not in body:
                raise ValueError(f"missing field {field!r}")
        if not isinstance(body["structure"], str):
            raise ValueError("structure must be a base64 string")
        if not isinstance(body["scene"], dict):
            raise ValueError("scene must be an object")
    width, height = int(body["width"]), int(body["height"])
    if width < 1 or height < 1:
        raise ValueError(f"size must be at least 1x1 pixels, got {width}x{height}")
    quality = body.get("quality", "fast")
    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"unknown quality {quality!r}, use one of {sorted(QUALITY_PRESETS)}"
        )
    return {
        "width": width,
        "height": height,
        "quality": quality,
        "transparent": bool(body.get("transparent", False)),
        "timeout": float(body.get("timeout", 300.0)),
    }


class RenderHandler(BaseHTTPRequestHandler):
    """Serve ``POST /render`` and ``GET /health``."""

    server_version = "moleculekit-render"

    def log_message(self, format, *args):  # noqa: A002 - signature is the base class's
        """Send request logs to the module logger rather than stderr."""
        logger.info("%s %s", self.address_string(), format % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - the base class names it this
        """Answer a health check."""
        if self.path.rstrip("/") != "/health":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, b'{"status": "ok"}', "application/json")

    def do_POST(self):  # noqa: N802 - the base class names it this
        """Render one image and return it as a PNG."""
        if self.path.rstrip("/") != "/render":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_REQUEST_BYTES:
            self._send(413, b"request too large", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(length))
            arguments = _validate(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, str(exc).encode(), "text/plain")
            return

        try:
            with _render_lock:
                png = render_png(body, **arguments)
        except Exception as exc:
            logger.exception("render failed")
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")
            return
        self._send(200, png, "image/png")


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve renders until interrupted.

    The browser is started before the first request and reused, so only the
    server start pays for it.

    Parameters
    ----------
    host : str, optional
        Interface to bind. The default keeps the service on this machine;
        use an address on a private network to serve other machines.
    port : int, optional
        Port to listen on.

    Raises
    ------
    RuntimeError
        If no browser can be started, since the service could serve nothing.
    """
    # Start the browser before the first client asks, not on its request.
    # Starting it is the expensive part, tens of seconds on a small container,
    # and it is then reused for the life of the process. Doing it here also
    # fails immediately and visibly when there is no usable browser, rather
    # than serving errors to whoever happens to connect first.
    _get_or_start(1200, 900)
    httpd = ThreadingHTTPServer((host, port), RenderHandler)
    logger.info("render server listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    """Run the render server from the command line."""
    parser = argparse.ArgumentParser(description="Serve moleculekit renders over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
