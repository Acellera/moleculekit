"""Headless Mol* rendering: turn a Molecule into a PNG with no display.

A singleton chromium is started on first use and reused, driven over the
devtools protocol (see cdp.py). The scene comes from build_mvs, the same
builder the interactive viewers use, so a render matches what view() shows.

Rendering runs on the GPU when one is reachable and falls back to a software
rasteriser when it is not, which is what lets the same code work inside
containers that expose no graphics device. MOLECULEKIT_RENDER_GL pins the
choice. The fallback is not free: a 1200x900 render measured 0.5s on a GPU
against 6.9s in software.

This module is not thread-safe: the singleton browser's devtools session has
one socket and one frame reader, so concurrent render() calls from different
threads can consume each other's responses. Call render() from one thread at
a time.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from moleculekit.viewer.molstar.cdp import WS, page_target_url

if TYPE_CHECKING:
    import numpy as np

    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)

_HEADLESS_PAGE = Path(__file__).parent / "static" / "headless.html"
_PORT_RANGE = range(9222, 9233)
_CHROMIUM_NAMES = ("chromium", "chromium-browser", "google-chrome", "chrome")
_PAGE_READY_POLL_INTERVAL = 0.05
_PAGE_READY_TIMEOUT = 30.0
_POSIX = os.name == "posix"

QUALITY_PRESETS = {
    "fast": {"occlusion": False},
    "high": {"occlusion": True},
}

_CHROMIUM_FLAGS = (
    "--headless=new",
    # The process already runs inside a container in every deployment we target,
    # and chromium's own sandbox cannot nest there.
    "--no-sandbox",
    # Docker's default /dev/shm is 64 MB, which chromium will happily exceed.
    "--disable-dev-shm-usage",
    # headless.html is loaded as a file:// URL, and each file:// document gets
    # a distinct opaque origin. Its <script type="module"> fetch is CORS-mode
    # regardless, and file is not among the schemes CORS mode permits, so the
    # module never loads and window.mkHeadless stays undefined. This flag is
    # the only thing that lets a file:// document import its own script. The
    # only document this browser ever loads is our own packaged bundle, and
    # the molecule's structure data reaches it as an inlined data: URL rather
    # than a file read, so the wider file access this grants is never used
    # for anything but our own static assets.
    "--allow-file-access-from-files",
    # Without these chromium spends about 25 seconds on Google service
    # registration before it draws anything.
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--metrics-recording-only",
    "--mute-audio",
)

# How chromium is told to reach OpenGL. The choice is worth roughly 15x on
# render time: a 1200x900 "fast" render measured 0.43s on an NVIDIA GPU and
# 6.9s on the software rasteriser, because SwiftShader fills every pixel on
# the CPU.
_GL_BACKENDS = {
    "hardware": ("--use-gl=angle", "--use-angle=gl"),
    "software": (
        "--use-gl=angle",
        "--use-angle=swiftshader",
        # Required from chromium 136 on: without it, WebGL context creation
        # fails outright when only the software rasteriser is available.
        "--enable-unsafe-swiftshader",
    ),
}
_GL_ENV_VAR = "MOLECULEKIT_RENDER_GL"
# A GPU chromium can reach appears here as a DRM render node.
_DRM_DIR = Path("/dev/dri")


@dataclass
class _RendererState:
    """A running headless browser and its devtools session."""

    process: subprocess.Popen
    ws: WS
    profile_dir: str
    gl_renderer: str
    gl_backend: str


def _hardware_gl_present() -> bool:
    """Whether this machine looks like it can render on a real GPU.

    On Linux a GPU reachable by chromium shows up as a DRM render node. The
    sandboxed containers moleculekit targets have none, so this is what keeps
    them from paying for a browser start that could never work. Elsewhere the
    check cannot be made this cheaply, so hardware is assumed and the launch
    itself decides.

    Returns
    -------
    present : bool
        True when a hardware GL attempt is worth making.
    """
    if not _POSIX:
        return True
    if not _DRM_DIR.is_dir():
        return False
    return any(node.name.startswith("renderD") for node in _DRM_DIR.iterdir())


def _gl_backend_order() -> list[str]:
    """The GL backends to try, in order, until one yields a WebGL context.

    ``MOLECULEKIT_RENDER_GL`` pins the choice to ``hardware`` or ``software``.
    Unset (or ``auto``) picks hardware first when a GPU looks reachable, and
    always keeps software as the fallback.

    Returns
    -------
    backends : list of str
        Keys of ``_GL_BACKENDS``, in the order they should be attempted.

    Raises
    ------
    ValueError
        If the environment variable names no known backend.
    """
    requested = os.environ.get(_GL_ENV_VAR, "auto").strip().lower()
    if requested in _GL_BACKENDS:
        return [requested]
    if requested != "auto":
        raise ValueError(
            f"{_GL_ENV_VAR}={requested!r} is not a known GL backend. Use one of "
            f"{sorted(_GL_BACKENDS)}, or 'auto'."
        )
    if _hardware_gl_present():
        return ["hardware", "software"]
    return ["software"]


def _usable_gl(gl_renderer: str | None) -> bool:
    """Whether an ``init()`` renderer string reports a working WebGL context."""
    if not gl_renderer:
        return False
    return "NO WEBGL" not in gl_renderer and "PROBE THREW" not in gl_renderer


_state: _RendererState | None = None
_state_lock = threading.Lock()


def _find_chromium_or_none() -> str | None:
    explicit = os.environ.get("MOLECULEKIT_CHROMIUM")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for name in _CHROMIUM_NAMES:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def find_chromium() -> str:
    """Locate a chromium or chrome binary to render with.

    Resolution order is the ``MOLECULEKIT_CHROMIUM`` environment variable, then
    ``chromium``, ``chromium-browser``, ``google-chrome`` and ``chrome`` on PATH.

    Returns
    -------
    path : str
        Path to the browser executable.

    Raises
    ------
    RuntimeError
        If no usable browser is found.
    """
    found = _find_chromium_or_none()
    if found is None:
        raise RuntimeError(
            "No chromium binary found for headless rendering. Install one "
            f"(any of {', '.join(_CHROMIUM_NAMES)}) or point the "
            "MOLECULEKIT_CHROMIUM environment variable at the executable."
        )
    return found


def _free_port() -> int:
    for port in _PORT_RANGE:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            sock.close()
        return port
    raise RuntimeError(
        f"No free port in {_PORT_RANGE.start}..{_PORT_RANGE.stop - 1} for the "
        "headless renderer devtools connection."
    )


def _wait_for_page_ready(ws: WS, timeout: float = _PAGE_READY_TIMEOUT) -> None:
    """Block until ``window.mkHeadless`` exists on the page.

    ``page_target_url`` only waits for a devtools *target* to appear, not for
    that target's document to finish loading its module script. Evaluating
    ``window.mkHeadless.init`` before the module has run raises a ``TypeError``
    for a property read on ``undefined``, so poll until it is defined.

    Parameters
    ----------
    ws : WS
        The devtools session for the headless page.
    timeout : float, optional
        Seconds to keep polling before giving up.

    Raises
    ------
    RuntimeError
        If ``window.mkHeadless`` never appears within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ws.evaluate("typeof window.mkHeadless !== 'undefined'"):
            return
        time.sleep(_PAGE_READY_POLL_INTERVAL)
    raise RuntimeError(
        f"headless.html did not finish loading within {timeout:g}s "
        "(window.mkHeadless never appeared)."
    )


def _kill_windows_tree(process: subprocess.Popen) -> None:
    """Best-effort process-tree kill on Windows.

    ``process.kill()`` alone calls ``TerminateProcess`` on only the tracked
    pid, leaving chromium's renderer, GPU and utility helpers running with
    open handles inside the profile directory. ``taskkill /T /F`` asks
    Windows to terminate the whole process tree instead. This path is
    untested from the development environment, which is Linux-only; if the
    ``taskkill`` call itself fails for any reason, fall back to the
    single-pid kill so cleanup still makes an attempt.

    Parameters
    ----------
    process : subprocess.Popen
        The chromium process (and its tree) to terminate.
    """
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        process.kill()


def _stop(process: subprocess.Popen, ws: WS | None, profile_dir: str) -> None:
    """Close the devtools session, kill the browser, and remove its profile dir.

    The wait for the process to actually exit must happen before the profile
    directory is removed: chromium's helper processes (GPU, renderer,
    crashpad) keep writing into it until they too have exited, and an rmtree
    that races that recreates files underneath it, leaving a corpse
    directory behind. Killing only the tracked pid is not enough to prevent
    that race: those helpers are not children of it in the ``wait()`` sense,
    so on POSIX ``_start`` launches chromium in its own new session
    (``start_new_session=True``, making it its process group leader) and
    this kills the whole group in one shot instead; on Windows
    ``_kill_windows_tree`` asks for the same thing via ``taskkill /T``.

    The removal of ``profile_dir`` always runs, even if the wait for the
    process to exit times out: a teardown failure must never propagate out
    and replace whatever exception the caller was already handling.

    Parameters
    ----------
    process : subprocess.Popen
        The chromium process to kill.
    ws : WS or None
        The devtools session to close, if one was ever opened.
    profile_dir : str
        The temporary profile directory to remove.
    """
    if ws is not None:
        try:
            ws.close()
        except OSError:
            pass
    if _POSIX:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        _kill_windows_tree(process)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    shutil.rmtree(profile_dir, ignore_errors=True)


def _start(width: int, height: int) -> _RendererState:
    """Start the browser, trying each GL backend until one gives a context.

    A hardware attempt that finds no GPU does not silently degrade: chromium
    comes up with no WebGL context at all, which is why each attempt is
    validated and the next one tried rather than assumed to work.
    """
    backends = _gl_backend_order()
    failures = []
    for backend in backends:
        state = _start_with_backend(width, height, backend)
        if state is not None:
            if backend != backends[0]:
                logger.debug(
                    "GL backend %r gave no WebGL context, fell back to %r",
                    backends[0],
                    backend,
                )
            return state
        failures.append(backend)
    raise RuntimeError(
        "Headless chromium started but no GL backend produced a WebGL context "
        f"(tried: {', '.join(failures)}). Renders would be blank. Set "
        f"{_GL_ENV_VAR} to force a specific backend."
    )


def _start_with_backend(
    width: int, height: int, backend: str
) -> _RendererState | None:
    """Start the browser on one GL backend. None when it yields no context."""
    binary = find_chromium()
    port = _free_port()
    profile_dir = tempfile.mkdtemp(prefix="moleculekit-render-")
    process = subprocess.Popen(
        [
            binary,
            *_CHROMIUM_FLAGS,
            *_GL_BACKENDS[backend],
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            f"--window-size={width},{height}",
            _HEADLESS_PAGE.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # New session (POSIX only) so this process is its own process group
        # leader: _stop() kills the whole group, taking chromium's helper
        # processes down with it instead of leaving them to linger.
        start_new_session=_POSIX,
    )
    ws: WS | None = None
    try:
        ws = WS(page_target_url(port))
        ws.call("Runtime.enable")
        _wait_for_page_ready(ws)
        gl_renderer = ws.evaluate(
            f"window.mkHeadless.init({int(width)}, {int(height)})"
        )
    except Exception:
        _stop(process, ws, profile_dir)
        raise
    if not _usable_gl(gl_renderer):
        _stop(process, ws, profile_dir)
        return None
    logger.debug(
        "headless renderer using %s GL, WebGL device: %s", backend, gl_renderer
    )
    return _RendererState(
        process=process,
        ws=ws,
        profile_dir=profile_dir,
        gl_renderer=gl_renderer,
        gl_backend=backend,
    )


def _get_or_start(width: int, height: int) -> _RendererState:
    global _state
    with _state_lock:
        if _state is not None and _state.process.poll() is not None:
            dead, _state = _state, None
            _stop(dead.process, dead.ws, dead.profile_dir)
        if _state is None:
            _state = _start(width, height)
        return _state


def shutdown_for_tests() -> None:
    """Stop the singleton browser if one is running."""
    global _state
    with _state_lock:
        state, _state = _state, None
        if state is not None:
            _stop(state.process, state.ws, state.profile_dir)


atexit.register(shutdown_for_tests)


def _png_dimensions(png: bytes) -> tuple[int, int]:
    """Read the width and height encoded in a PNG's IHDR chunk.

    Parameters
    ----------
    png : bytes
        A complete PNG file.

    Returns
    -------
    size : tuple of int
        The image width and height in pixels, as encoded in the file.
    """
    width, height = struct.unpack(">II", png[16:24])
    return width, height


def render(
    mol: "Molecule",
    output: str | None = None,
    *,
    size: tuple[int, int] = (1200, 900),
    quality: str = "fast",
    center: "str | np.ndarray | None" = None,
    rotate=None,
    zoom: float | None = None,
    background: str = "white",
    transparent: bool = False,
    timeout: float = 300.0,
) -> bytes | str:
    """Render ``mol`` to a PNG with no display and no browser window.

    The scene is the one ``view()`` would show: representations come from
    ``mol.reps``, and the frame rendered is ``mol.frame``.

    Parameters
    ----------
    mol : Molecule
        The molecule to render.
    output : str or None, optional
        Path to write the PNG to. When None the PNG bytes are returned.
    size : tuple of int, optional
        Image width and height in pixels.
    quality : str, optional
        One of the keys of ``QUALITY_PRESETS``. ``"high"`` enables ambient
        occlusion, which is close to free on a GPU and costs roughly three
        times the render time on the software fallback.
    center : str or np.ndarray or None, optional
        Atom selection to frame the camera on. None frames the whole structure.
    rotate : str or tuple of float or None, optional
        Camera orientation, as a preset name or ``(rx, ry, rz)`` in degrees.
    zoom : float or None, optional
        Camera tightness. Larger values move the camera closer.
    background : str, optional
        Background colour as an SVG colour name or hex string.
    transparent : bool, optional
        Render onto a transparent background, ignoring ``background``.
    timeout : float, optional
        Seconds to allow for a single render before giving up.

    Returns
    -------
    result : bytes or str
        The PNG bytes when ``output`` is None, otherwise ``output``.

    Raises
    ------
    ValueError
        If ``quality`` names no known preset, if ``size`` is not at least one
        pixel in each dimension, if ``center`` matches no atoms, if ``zoom``
        is not positive, or if ``rotate`` is a string that names no known
        orientation preset.
    RuntimeError
        If no browser is found, if WebGL is unavailable, if the page fails,
        or if the rendered image does not match the requested size.
    """
    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"Unknown quality {quality!r}. Use one of {sorted(QUALITY_PRESETS)}."
        )
    width, height = int(size[0]), int(size[1])
    if width < 1 or height < 1:
        # Mol*'s draw() early-returns on a non-positive canvas size before
        # touching the canvas, and the screenshot helper then serializes
        # whatever the canvas held from before: a plausible-looking image of
        # the wrong (or no) scene rather than an error. Reject it here.
        raise ValueError(f"size must be at least 1x1 pixels, got {size!r}.")
    if zoom is not None and zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom!r}.")
    if center is not None and not mol.atomselect(center).any():
        # build_mvs treats a non-matching focus_sel as a no-op, which is the
        # right contract for that lower-level, permissive builder. render()
        # is public API though, so a center that silently produces a
        # default-orientation image instead of the requested one must raise.
        raise ValueError(f"center selection matched no atoms: {center!r}")

    from moleculekit.viewer.molstar.inline import (
        _b64,
        _bcif_bytes,
        _scene_from_reps,
    )
    from moleculekit.viewer.molstar.mvs import build_mvs

    scene = _scene_from_reps(mol, mol.reps.replist + mol._tempreps.replist)
    structure_url = "data:application/octet-stream;base64," + _b64(_bcif_bytes(mol))
    mvsj = build_mvs(
        mol,
        structure_url=structure_url,
        representations=scene.get("representations") or None,
        focus_sel=center,
        rotate=rotate,
        zoom=zoom,
        background_color=None if transparent else background,
    )

    state = _get_or_start(width, height)
    # The devtools session's own read timeout was fixed at 300s when the
    # singleton browser was started (cdp.WS's default). Widen it to this
    # call's timeout so a caller-requested value above 300s can actually
    # take effect, rather than a bare socket.timeout firing first and
    # masking the render timeout this function documents.
    state.ws.set_timeout(timeout)
    timeout_ms = int(timeout * 1000)
    try:
        state.ws.evaluate(
            f"window.mkHeadless.load({json.dumps(mvsj)})", timeout_ms=timeout_ms
        )
        options = {
            "width": width,
            "height": height,
            "occlusion": QUALITY_PRESETS[quality]["occlusion"],
            "transparent": bool(transparent),
        }
        data_uri = state.ws.evaluate(
            f"window.mkHeadless.screenshot({json.dumps(options)})",
            timeout_ms=timeout_ms,
        )
    except Exception:
        # A wedged browser must not be inherited by the next call.
        shutdown_for_tests()
        raise

    png = base64.b64decode(data_uri.split(",", 1)[1])
    actual_size = _png_dimensions(png)
    if actual_size != (width, height):
        # Mol*'s canvas silently keeps a stale frame when draw() early-returns
        # (see the size validation above); catching a size mismatch here is
        # the general form of that guard, independent of what caused it.
        raise RuntimeError(
            f"Screenshot returned a {actual_size[0]}x{actual_size[1]} image "
            f"but {width}x{height} was requested; the render likely failed "
            "silently."
        )
    if output is None:
        return png
    with open(output, "wb") as fh:
        fh.write(png)
    return output
