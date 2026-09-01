import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar import render as render_mod
from moleculekit.viewer.molstar import renderserver


def _mol():
    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)
    return mol


class _Server:
    """A render server on a free port, for the duration of a test."""

    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), renderserver.RenderHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=10)


def test_validate_accepts_a_well_formed_request():
    body = {"structure": "eA==", "scene": {"components": []}, "width": 10, "height": 8}
    assert renderserver._validate(body) == {
        "width": 10,
        "height": 8,
        "quality": "fast",
        "transparent": False,
        "timeout": 300.0,
    }


@pytest.mark.parametrize(
    "body,message",
    [
        ({"scene": {}, "width": 1, "height": 1}, "missing field 'structure'"),
        ({"structure": "x", "width": 1, "height": 1}, "missing field 'scene'"),
        ({"structure": 1, "scene": {}, "width": 1, "height": 1}, "base64 string"),
        ({"structure": "x", "scene": [], "width": 1, "height": 1}, "must be an object"),
        ({"structure": "x", "scene": {}, "width": 0, "height": 1}, "at least 1x1"),
        (
            {"structure": "x", "scene": {}, "width": 1, "height": 1, "quality": "epic"},
            "unknown quality",
        ),
    ],
)
def test_validate_rejects_bad_requests(body, message):
    with pytest.raises(ValueError, match=message):
        renderserver._validate(body)


def test_health_endpoint():
    with _Server() as url:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
            assert json.load(response) == {"status": "ok"}


def test_unknown_paths_are_not_found():
    with _Server() as url:
        for path, data in (("/nope", None), ("/nope", b"{}")):
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(f"{url}{path}", data=data, timeout=10)
            assert excinfo.value.code == 404


def test_render_with_a_server_never_starts_a_local_browser(monkeypatch):
    """The point of a render server is that the client needs no browser."""
    monkeypatch.setattr(
        render_mod,
        "_get_or_start",
        lambda *a: pytest.fail("render() started a local browser despite server="),
    )
    monkeypatch.setattr(renderserver, "render_png", lambda *a, **k: b"\x89PNG-stub")

    with _Server() as url:
        assert render_mod.render(_mol(), server=url) == b"\x89PNG-stub"


def test_the_server_receives_what_the_client_built(monkeypatch):
    seen = {}

    def _fake(payload, **kwargs):
        seen["structure"] = payload["structure"]
        seen["scene"] = payload["scene"]
        seen["kwargs"] = kwargs
        return b"png"

    monkeypatch.setattr(renderserver, "render_png", _fake)
    with _Server() as url:
        render_mod.render(_mol(), size=(321, 123), quality="high", server=url)

    assert seen["kwargs"] == {
        "width": 321,
        "height": 123,
        "quality": "high",
        "transparent": False,
        "timeout": 300.0,
    }
    assert seen["scene"]["components"], "the scene was built client-side and sent"
    assert isinstance(seen["structure"], str) and seen["structure"]


def test_a_failing_render_comes_back_as_an_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no WebGL for you")

    monkeypatch.setattr(renderserver, "render_png", _boom)
    with _Server() as url:
        with pytest.raises(RuntimeError, match="no WebGL for you"):
            render_mod.render(_mol(), server=url)


def test_an_unreachable_server_says_so():
    with pytest.raises(RuntimeError, match="Cannot reach render server"):
        render_mod.render(_mol(), server="http://127.0.0.1:1")


def test_the_environment_variable_selects_a_server(monkeypatch):
    monkeypatch.setattr(renderserver, "render_png", lambda *a, **k: b"env-png")
    monkeypatch.setattr(
        render_mod,
        "_get_or_start",
        lambda *a: pytest.fail("the environment variable was ignored"),
    )
    with _Server() as url:
        monkeypatch.setenv(render_mod._SERVER_ENV_VAR, url)
        assert render_mod.render(_mol()) == b"env-png"


@pytest.mark.skipif(
    render_mod._find_chromium_or_none() is None, reason="needs a chromium binary"
)
def test_a_served_render_matches_a_local_one():
    """End to end: the same molecule, drawn here and drawn through the server."""
    from pathlib import Path

    mol = Molecule(str(Path(__file__).parent / "test_molecule" / "3ptb_filtered.pdb"))
    mol.filter("not water")
    local = render_mod.render(mol, size=(200, 150))
    with _Server() as url:
        served = render_mod.render(mol, size=(200, 150), server=url)
    assert served == local
    render_mod.shutdown_for_tests()


def test_image_bytes_survive_the_round_trip(monkeypatch):
    """Binary must not be mangled by the HTTP layer."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (7, 5), (10, 200, 30)).save(buf, format="PNG")
    monkeypatch.setattr(renderserver, "render_png", lambda *a, **k: buf.getvalue())
    with _Server() as url:
        got = render_mod.render(_mol(), server=url)
    assert got == buf.getvalue()
    assert Image.open(io.BytesIO(got)).size == (7, 5)


def test_serve_starts_the_browser_before_accepting_requests(monkeypatch):
    """The browser start is the expensive part, so a client must never pay it.

    It is also the failure worth surfacing at startup: a machine with no usable
    browser should not accept a port and then serve errors.
    """
    started = []
    monkeypatch.setattr(renderserver, "_get_or_start", lambda *a: started.append(a))

    class _FakeHTTPD:
        def __init__(self, *a):
            self.address = a

        def serve_forever(self):
            assert started, "serve_forever ran before the browser was started"

        def server_close(self):
            pass

    monkeypatch.setattr(renderserver, "ThreadingHTTPServer", _FakeHTTPD)
    renderserver.serve("127.0.0.1", 0)
    assert started, "the browser was never started"


def test_the_server_accepts_several_objects(monkeypatch):
    """A figure of objects loaded separately has to survive the wire.

    One object keeps the older single-structure request shape, so a render
    server on an older moleculekit still answers the common case.
    """
    seen = {}

    def _capture(payload, **kwargs):
        seen["p"] = payload
        return b"png"

    monkeypatch.setattr(renderserver, "render_png", _capture)
    a, b = _mol(), _mol()
    a.reps.add("all", "Licorice", "Name")
    b.reps.add("all", "VDW", "Name")

    with _Server() as url:
        assert render_mod.render([a, b], server=url) == b"png"
    assert len(seen["p"]["objects"]) == 2
    assert "globals" in seen["p"]


@pytest.mark.parametrize(
    "body,message",
    [
        ({"objects": [], "width": 1, "height": 1}, "non-empty list"),
        ({"objects": [{"scene": {}}], "width": 1, "height": 1}, "base64 structure"),
        ({"objects": [{"structure": "x"}], "width": 1, "height": 1}, "needs a scene"),
    ],
)
def test_bad_multi_object_requests_are_rejected(body, message):
    with pytest.raises(ValueError, match=message):
        renderserver._validate(body)
