"""Headless Mol* rendering: turn a Molecule into a PNG with no display.

A singleton chromium is started on first use and reused, driven over the
devtools protocol (see cdp.py). The scene comes from build_scene (see
scene.py), the same description the interactive viewer builds from, and the
browser applies it with applyScene (see viewer-frontend/src/scene.ts) so a
render matches what view() shows. This path never imports molviewspec.

Rendering runs on the GPU when one is reachable and falls back to a software
rasteriser when it is not, which is what lets the same code work inside
containers that expose no graphics device. MOLECULEKIT_RENDER_GL pins the
choice. The fallback is not free: a 1200x900 "fast" render measured 0.16s on a
GPU against 1.2s on a 20-core software rasteriser, and software time scales
with pixels and with how many cores it has.

This module is not thread-safe: the singleton browser's devtools session has
one socket and one frame reader, so concurrent render() calls from different
threads can consume each other's responses. Call render() from one thread at
a time.
"""

from __future__ import annotations

import atexit
import base64
import gzip
import json
import logging
import os
import shutil
import signal
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
_CHROMIUM_NAMES = ("chromium", "chromium-browser", "google-chrome", "chrome")
_PAGE_READY_POLL_INTERVAL = 0.05
_PAGE_READY_TIMEOUT = 30.0
_POSIX = os.name == "posix"

# sample_level is the anti-aliasing work: the scene is drawn 2**level times
# and accumulated. Mol*'s own screenshot helper hardcodes 4, so 16 renders,
# which costs nothing on a GPU and dominates the time on the software
# rasteriser. "fast" spends 2 renders on it instead.
QUALITY_PRESETS = {
    "fast": {"occlusion": False, "sample_level": 1},
    "high": {"occlusion": True, "sample_level": 4},
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

# How chromium is told to reach OpenGL. The choice is worth roughly 7x on
# render time: a 1200x900 "fast" render measured 0.16s on an NVIDIA GPU and
# 1.2s on the software rasteriser, because SwiftShader fills every pixel on
# the CPU, across as many cores as it has.
_VULKAN_FLAGS = (
    "--use-gl=angle",
    "--use-angle=vulkan",
    # Chromium refuses a Vulkan device without these, and then reports no
    # WebGL context at all rather than saying why.
    "--ignore-gpu-blocklist",
    "--enable-features=Vulkan",
)

_GL_BACKENDS = {
    # ANGLE on Vulkan against a real GPU. Unlike the GL backend this needs no
    # display of any kind, which is what makes GPU rendering possible in a
    # container: no X server, no socket, no cookie, just the GPU. Measured on a
    # GPU-less-looking container with the GPU passed in, 1M63 at 1400x1000 took
    # 3.7s against 28.7s in software.
    "hardware-vulkan": _VULKAN_FLAGS + ("--disable-gpu-sandbox",),
    "hardware": (
        "--use-gl=angle",
        "--use-angle=gl",
        # Without this the GPU process cannot open the driver's device nodes
        # inside a container: chromium starts, the GPU process starts, and the
        # page still reports no WebGL context, with nothing in the log saying
        # why. The browser only ever loads our own packaged page, with no
        # network and no user content, so the sandbox it gives up guards
        # nothing here.
        "--disable-gpu-sandbox",
    ),
    # Mesa's software Vulkan (lavapipe). Measured about 30% faster than
    # SwiftShader on the same GPU-less container: 1M63 at 1000x750 took a
    # median 1.7s against 2.4s. Only the Vulkan path works headless, because
    # ANGLE's GL backend wants an X display and says so ("Could not open the
    # default X display"), so llvmpipe cannot be reached through OpenGL here.
    "software-vulkan": _VULKAN_FLAGS,
    "software": (
        "--use-gl=angle",
        "--use-angle=swiftshader",
        # Required from chromium 136 on: without it, WebGL context creation
        # fails outright when only the software rasteriser is available.
        "--enable-unsafe-swiftshader",
    ),
}
_GL_ENV_VAR = "MOLECULEKIT_RENDER_GL"
# Points render() at a render server instead of a local browser, so a machine
# with no GPU (or no chromium at all) can still render.
_SERVER_ENV_VAR = "MOLECULEKIT_RENDER_SERVER"
# A GPU chromium can reach appears here as a DRM render node.
_DRM_DIR = Path("/dev/dri")
# PCI vendor of a DRM render node, and the Vulkan driver manifests that vendor
# installs. Used to pin one driver: handing ANGLE every manifest on the machine
# makes it fail outright, since most of them are for hardware that is not here.
_PCI_VENDOR_ICDS = {
    "0x10de": ("nvidia",),
    "0x1002": ("radeon", "amd"),
    "0x8086": ("intel",),
}
_DRM_CLASS_DIR = Path("/sys/class/drm")

# Where Linux advertises Vulkan drivers. Distributions use the first; the
# NVIDIA container toolkit drops its manifest in the second.
_VULKAN_ICD_DIRS = (Path("/usr/share/vulkan/icd.d"), Path("/etc/vulkan/icd.d"))


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


def _vulkan_manifests():
    """Every Vulkan driver manifest installed here.

    Returns
    -------
    manifests : list of Path
        The manifest files, in a stable order.
    """
    found = []
    for directory in _VULKAN_ICD_DIRS:
        if directory.is_dir():
            found.extend(sorted(directory.glob("*.json")))
    return found


def _software_vulkan_icd() -> str | None:
    """Mesa's software Vulkan driver (lavapipe), when it is installed.

    Returns
    -------
    icd : str or None
        Path to the manifest, or None when Mesa's Vulkan drivers are absent
        (Debian and Ubuntu ship them in mesa-vulkan-drivers).
    """
    for icd in _vulkan_manifests():
        if icd.name.startswith("lvp_"):
            return str(icd)
    return None


def _hardware_vulkan_icds() -> list[str]:
    """Vulkan drivers worth trying, for GPUs this machine actually has.

    One driver is pinned per attempt rather than all of them at once: ANGLE
    fails outright when handed manifests for hardware that is not present.
    Pinning matters in the other direction too, since with Mesa's software
    driver visible ANGLE picks it and renders at software speed while
    reporting success. Which of several candidates works cannot be known
    without trying, so they come back in preference order: a machine with
    both an Intel display adapter and an NVIDIA card should render on the
    NVIDIA one.

    Linux only, which is where this matters. Elsewhere this finds nothing and
    rendering falls back to the paths that need no driver installed.

    Returns
    -------
    icds : list of str
        Manifest paths, best first, empty when no GPU here has one.
    """
    vendors = set()
    for node in sorted(_DRM_DIR.glob("renderD*")) if _DRM_DIR.is_dir() else []:
        try:
            vendor = (_DRM_CLASS_DIR / node.name / "device/vendor").read_text()
        except OSError:
            continue
        vendors.add(vendor.strip().lower())

    icds = []
    for vendor, prefixes in _PCI_VENDOR_ICDS.items():
        if vendor not in vendors:
            continue
        for prefix in prefixes:
            # Shortest name first, so intel_icd wins over intel_hasvk_icd, the
            # legacy driver for hardware a decade older than this code.
            matches = [i for i in _vulkan_manifests() if i.name.startswith(prefix)]
            icds.extend(str(i) for i in sorted(matches, key=lambda i: len(i.name)))
    return icds


def _gl_backend_order() -> list[str]:
    """The GL backends to try, in order, until one yields a WebGL context.

    ``MOLECULEKIT_RENDER_GL`` pins the choice to one of ``_GL_BACKENDS``.
    Unset (or ``auto``) picks hardware first when a GPU looks reachable, then
    Mesa's software Vulkan where it is installed, and always keeps SwiftShader
    as the fallback since it needs nothing installed.

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
    order = []
    if _hardware_gl_present():
        # Vulkan first: it draws on the GPU with or without a display, where
        # the GL backend needs one and gets no WebGL context without it.
        if _hardware_vulkan_icds():
            order.append("hardware-vulkan")
        order.append("hardware")
    if _software_vulkan_icd() is not None:
        order.append("software-vulkan")
    # SwiftShader last and always: it needs nothing installed.
    order.append("software")
    return order


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


def _devtools_port(profile_dir: str, process: subprocess.Popen) -> int:
    """Read the devtools port chromium chose, from its profile directory.

    Picking a free port here instead and passing it in cannot be done safely:
    the probe socket has to be closed before chromium can bind it, so two
    processes starting a renderer at the same time both see the same port
    free and the second one connects to the first one's browser. They then
    clear each other's Mol* state ("Could not find node") and close each
    other's browser ("websocket closed by peer"), which is what running the
    test suite under pytest-xdist did. Letting chromium bind port 0 and
    reporting back removes the window entirely, and the profile directory is
    already unique per process.

    Parameters
    ----------
    profile_dir : str
        The ``--user-data-dir`` chromium was started with.
    process : subprocess.Popen
        The chromium process, watched so a crash is reported as one.

    Returns
    -------
    port : int
        The port chromium is listening on.

    Raises
    ------
    RuntimeError
        If chromium exits, or does not report a port in time.
    """
    port_file = Path(profile_dir) / "DevToolsActivePort"
    deadline = time.time() + _PAGE_READY_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"The headless browser exited with code {process.returncode} "
                "before reporting a devtools port."
            )
        try:
            # First line is the port, second the browser's websocket path.
            first = port_file.read_text().splitlines()[0]
        except (OSError, IndexError):
            time.sleep(_PAGE_READY_POLL_INTERVAL)
            continue
        return int(first)
    raise RuntimeError(
        f"The headless browser did not report a devtools port within "
        f"{_PAGE_READY_TIMEOUT} seconds."
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
    if backend == "hardware-vulkan":
        # Pin one driver per attempt and take the first that draws. Handing
        # ANGLE several at once fails, and leaving it unpinned lets it pick
        # Mesa's software driver and report success at software speed.
        for icd in _hardware_vulkan_icds():
            state = _launch(width, height, backend, {**os.environ, "VK_ICD_FILENAMES": icd})
            if state is not None:
                return state
        return None
    env = None
    if backend == "software-vulkan":
        env = {**os.environ, "VK_ICD_FILENAMES": _software_vulkan_icd() or ""}
    return _launch(width, height, backend, env)


def _launch(width: int, height: int, backend: str, env: dict | None):
    """Start one browser on one backend.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    backend : str
        The key of ``_GL_BACKENDS`` to start with.
    env : dict or None
        Environment for the browser, or None to inherit this process's.

    Returns
    -------
    state : _RendererState or None
        The running browser, or None when it yields no WebGL context.
    """
    binary = find_chromium()
    profile_dir = tempfile.mkdtemp(prefix="moleculekit-render-")
    process = subprocess.Popen(
        [
            binary,
            *_CHROMIUM_FLAGS,
            *_GL_BACKENDS[backend],
            # 0 means "pick a free port and write it to DevToolsActivePort".
            "--remote-debugging-port=0",
            f"--user-data-dir={profile_dir}",
            f"--window-size={width},{height}",
            _HEADLESS_PAGE.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        # New session (POSIX only) so this process is its own process group
        # leader: _stop() kills the whole group, taking chromium's helper
        # processes down with it instead of leaving them to linger.
        start_new_session=_POSIX,
    )
    ws: WS | None = None
    try:
        port = _devtools_port(profile_dir, process)
        ws = WS(page_target_url(port))
        ws.call("Runtime.enable")
        _wait_for_page_ready(ws)
        gl_renderer = ws.evaluate(
            f"window.mkHeadless.init({int(width)}, {int(height)})"
        )
    except Exception as exc:
        _stop(process, ws, profile_dir)
        # A backend that cannot make a WebGL context is a backend to skip, not
        # a failed render. init() reports that by throwing rather than by
        # returning a renderer string, so without this a machine that looks
        # like it has a GPU but cannot reach it never falls back to software:
        # mounting an NVIDIA GPU into a headless container puts a render node
        # in /dev/dri, which is all _hardware_gl_present() looks for, while
        # ANGLE still has no display to draw through.
        if "NO WEBGL" in str(exc) or "could not initialise" in str(exc).lower():
            logger.debug("%s GL yielded no WebGL context, trying the next", backend)
            return None
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


def _default_volume_rep(vol):
    """One isosurface for a volume the caller styled no further."""
    from moleculekit.volume import _VolumeRepresentation

    return _VolumeRepresentation(isovalue=vol.suggest_isovalue())


def _multi_payload(mols, *, center, rotate, zoom, background, transparent, fog, clip):
    """Per-object scenes plus the parts that belong to the whole picture.

    Each object keeps its own representations, so a protein and a ligand loaded
    separately stay separate: merging them into one structure first would lose
    which atoms belong to which. The camera and canvas are shared, and the
    camera's sphere is worked out here because a selection spanning several
    structures cannot be expressed as atom indices into one of them.

    Parameters
    ----------
    mols : list of Molecule
        The objects to draw, in order.
    center : str or np.ndarray or None
        Atom selection to frame, resolved against every object.
    rotate : str or tuple of float or None
        Camera orientation.
    zoom : float or None
        Camera tightness.
    background : str
        Background colour.
    transparent : bool
        Whether to render onto a transparent background.
    fog : float or None
        Depth cueing strength.
    clip : float or None
        Slab half-thickness.

    Returns
    -------
    scenes : list of dict
        One scene per object, carrying its components and labels.
    globals : dict
        The camera and canvas for the picture as a whole.
    """
    from moleculekit.viewer.molstar.scene import build_scene, focus_sphere

    scenes = []
    for index, mol in enumerate(mols):
        try:
            scenes.append(build_scene(mol, mol.reps.replist + mol._tempreps.replist))
        except ValueError as exc:
            # Say which object: with several of them, "every representation
            # matched no atoms" gives no clue which one to look at, and one
            # bad object takes the whole picture down.
            name = getattr(mol, "viewname", None)
            where = f"object {index}" + (f" ({name})" if name else "")
            raise ValueError(f"{where}: {exc}") from exc

    globals_: dict = {"components": []}
    canvas: dict = {}
    if not transparent and background is not None:
        canvas["background"] = background
    if fog is not None:
        canvas["fog"] = float(fog)
    if canvas:
        globals_["canvas"] = canvas

    if rotate is not None or zoom is not None or center is not None:
        from moleculekit.viewer.molstar.scene import rotation_to_direction_up

        focus_center, focus_radius = focus_sphere(mols, center)
        _, scene_radius = focus_sphere(mols)
        radius = focus_radius * (1.0 / float(zoom) if zoom is not None else 1.0)
        camera = {
            "center": focus_center,
            "radius": radius,
            # Without a slab, the planes take in the whole scene rather than
            # only what is framed, so focusing one object does not cut the
            # others away.
            "clip_radius": float(clip) if clip is not None else max(radius, scene_radius),
        }
        if rotate is not None or zoom is not None:
            direction, up = rotation_to_direction_up(rotate)
            camera["direction"] = list(direction)
            camera["up"] = list(up)
        globals_["camera"] = camera
    return scenes, globals_


def _scene_description(
    mol: "Molecule",
    *,
    center: "str | np.ndarray | None" = None,
    rotate=None,
    zoom: float | None = None,
    background: str = "white",
    transparent: bool = False,
    fog: float | None = None,
    clip: float | None = None,
) -> dict:
    """Build the scene description ``render()`` sends to the browser.

    Representations come from ``mol.reps`` together with any one-off
    representation added by ``view()``'s ``sel``/``style``/``color``
    arguments, exactly as the interactive viewer's ``_topology_event``
    builds its scene, so the two stay in lockstep.

    Parameters
    ----------
    mol : Molecule
        The molecule to describe.
    center : str or np.ndarray or None, optional
        Atom selection to frame the camera on. None frames the whole structure.
    rotate : str or tuple of float or None, optional
        Camera orientation, as a preset name or ``(rx, ry, rz)`` in degrees.
    zoom : float or None, optional
        Camera tightness. Larger values move the camera closer.
    background : str, optional
        Background colour as an SVG colour name or hex string.
    transparent : bool, optional
        Ignore ``background`` and render onto a transparent background.
    fog : float or None, optional
        Depth cueing strength, from 0 for none to 100 for the strongest.
    clip : float or None, optional
        Half-thickness in Angstrom of the slab drawn around what the camera
        frames. None draws the whole structure.

    Returns
    -------
    description : dict
        The scene description, as produced by
        :func:`moleculekit.viewer.molstar.scene.build_scene`.
    """
    from moleculekit.viewer.molstar.scene import build_scene

    return build_scene(
        mol,
        mol.reps.replist + mol._tempreps.replist,
        focus_sel=center,
        rotate=rotate,
        zoom=zoom,
        background_color=None if transparent else background,
        fog=fog,
        clip=clip,
    )


def render(
    mol,
    output: str | None = None,
    *,
    size: tuple[int, int] = (1200, 900),
    quality: str = "fast",
    center: "str | np.ndarray | None" = None,
    rotate=None,
    zoom: float | None = None,
    background: str = "white",
    transparent: bool = False,
    fog: float | None = None,
    clip: float | None = None,
    server: str | None = None,
    timeout: float = 300.0,
) -> bytes | str:
    """Render ``mol`` to a PNG with no display and no browser window.

    The scene is the one ``view()`` would show: representations come from
    ``mol.reps``, and the frame rendered is ``mol.frame``.

    Parameters
    ----------
    mol : Molecule or Volume or list
        The object to render, or several to draw together in one picture. Each
        keeps its own representations, so objects loaded separately stay
        separate. A :class:`moleculekit.volume.Volume` draws its isosurfaces
        alongside the molecules; at least one molecule is needed, since a
        volume has no atoms for the camera to frame.
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
    fog : float or None, optional
        Depth cueing strength, from 0 for none to 100 for the strongest. Fog
        fades distant geometry into the background colour, which reads as
        depth on a crowded structure. None uses Mol*'s own strength.
    clip : float or None, optional
        Half-thickness in Angstrom of the slab drawn around what the camera
        frames: geometry nearer to or further from the camera than this is cut
        away, which is how you see into a buried pocket. None draws the whole
        structure.
    server : str or None, optional
        Base URL of a render server, such as ``"http://gpuhost:8080"``, to draw
        the image on instead of starting a browser here. None falls back to the
        ``MOLECULEKIT_RENDER_SERVER`` environment variable, and then to
        rendering locally.
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
        is not positive, if ``rotate`` is a string that names no known
        orientation preset, if ``fog`` falls outside 0 to 100, or if ``clip``
        is not positive.
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
    from moleculekit.volume import Volume

    objects_in = list(mol) if isinstance(mol, (list, tuple)) else [mol]
    if not objects_in:
        raise ValueError("render() needs at least one object to draw.")
    volumes = [o for o in objects_in if isinstance(o, Volume)]
    mols = [o for o in objects_in if not isinstance(o, Volume)]
    if not mols:
        raise ValueError(
            "render() needs at least one molecule: a volume has no atoms, so "
            "on its own there is nothing to frame the camera on."
        )
    if center is not None and not any(m.atomselect(center).any() for m in mols):
        # build_scene treats a non-matching focus_sel as a no-op, which is the
        # right contract for that lower-level, permissive builder. render()
        # is public API though, so a center that silently produces a
        # default-orientation image instead of the requested one must raise.
        raise ValueError(f"center selection matched no atoms: {center!r}")

    from moleculekit.viewer.molstar.inline import _b64, _bcif_bytes

    multi = len(mols) > 1 or bool(volumes)
    if multi:
        scenes, globals_ = _multi_payload(
            mols,
            center=center,
            rotate=rotate,
            zoom=zoom,
            background=background,
            transparent=transparent,
            fog=fog,
            clip=clip,
        )
        objects = [
            {"structure": _b64(_bcif_bytes(m)), "scene": scene}
            for m, scene in zip(mols, scenes)
        ]
        if volumes:
            globals_["volumes"] = [
                {
                    # gzipped because base64 of a 200-cell grid is 43MB on
                    # its own. Level 1: float32 mantissa bits are close to
                    # random, so a density gains 1.1x to 2x and no level
                    # spends its way past that.
                    "ccp4_gz": _b64(gzip.compress(vol.to_ccp4(), 1)),
                    "reps": [
                        {
                            "isovalue": float(rep.isovalue),
                            "color": rep.color,
                            "opacity": float(rep.opacity),
                            "wireframe": bool(rep.wireframe),
                        }
                        # A volume with no representations set gets one surface
                        # at a value taken from its own data.
                        for rep in (vol.reps.replist or [_default_volume_rep(vol)])
                        if rep.visibility
                    ],
                }
                for vol in volumes
            ]
    else:
        description = _scene_description(
            mols[0],
            center=center,
            rotate=rotate,
            zoom=zoom,
            background=background,
            transparent=transparent,
            fog=fog,
            clip=clip,
        )
        bcif_b64 = _b64(_bcif_bytes(mols[0]))

    if server is None:
        server = os.environ.get(_SERVER_ENV_VAR) or None
    # One object keeps the single-object wire format, so a render server on an
    # older moleculekit still answers the common case.
    payload = (
        {"objects": objects, "globals": globals_}
        if multi
        else {"structure": bcif_b64, "scene": description}
    )
    if server:
        png = _render_remote(
            server,
            payload,
            width=width,
            height=height,
            quality=quality,
            transparent=transparent,
            timeout=timeout,
        )
    else:
        png = render_png(
            payload,
            width=width,
            height=height,
            quality=quality,
            transparent=transparent,
            timeout=timeout,
        )
    if output is None:
        return png
    with open(output, "wb") as fh:
        fh.write(png)
    return output


def _render_remote(
    server: str,
    payload: dict,
    *,
    width: int,
    height: int,
    quality: str,
    transparent: bool,
    timeout: float,
) -> bytes:
    """Ask a render server for the image instead of drawing it here.

    Parameters
    ----------
    server : str
        Base URL of the render server.
    payload : dict
        What to draw: one structure and its scene, or several objects and the
        camera and canvas they share.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    quality : str
        One of the keys of ``QUALITY_PRESETS``.
    transparent : bool
        Whether to render onto a transparent background.
    timeout : float
        Seconds to allow before giving up.

    Returns
    -------
    png : bytes
        The rendered image.

    Raises
    ------
    RuntimeError
        If the server rejects the request or fails to render it.
    """
    import urllib.error
    import urllib.request

    body = json.dumps(
        {
            **payload,
            "width": width,
            "height": height,
            "quality": quality,
            "transparent": bool(transparent),
            "timeout": timeout,
        }
    ).encode()
    request = urllib.request.Request(
        server.rstrip("/") + "/render",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        # The server has to finish the render before it answers, so allow it
        # the caller's own budget plus a little for the round trip.
        with urllib.request.urlopen(request, timeout=timeout + 30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Render server {server} refused the render: {exc.read().decode()[:400]}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot reach render server {server}: {exc}") from exc


def render_png(
    payload: dict,
    *,
    width: int,
    height: int,
    quality: str = "fast",
    transparent: bool = False,
    timeout: float = 300.0,
) -> bytes:
    """Draw one already-built scene in the local browser.

    This is the half of :func:`render` that needs a browser. Everything above
    it, turning a Molecule into a structure and a scene description, is plain
    Python, which is what lets a render run on another machine: the render
    server (see renderserver.py) calls this with what a client sent it.

    Parameters
    ----------
    payload : dict
        What to draw: ``structure`` and ``scene`` for one object, or
        ``objects`` and ``globals`` for several drawn together.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    quality : str, optional
        One of the keys of ``QUALITY_PRESETS``.
    transparent : bool, optional
        Render onto a transparent background.
    timeout : float, optional
        Seconds to allow before giving up.

    Returns
    -------
    png : bytes
        The rendered image.

    Raises
    ------
    RuntimeError
        If the rendered image does not match the requested size.
    """
    state = _get_or_start(width, height)
    # The devtools session's own read timeout was fixed at 300s when the
    # singleton browser was started (cdp.WS's default). Widen it to this
    # call's timeout so a caller-requested value above 300s can actually
    # take effect, rather than a bare socket.timeout firing first and
    # masking the render timeout this function documents.
    state.ws.set_timeout(timeout)
    timeout_ms = int(timeout * 1000)
    try:
        objects = payload.get("objects")
        structures = (
            [o["structure"] for o in objects]
            if objects is not None
            else [payload["structure"]]
        )
        state.ws.evaluate(
            f"window.mkHeadless.loadStructures({json.dumps(structures)})",
            timeout_ms=timeout_ms,
        )
        volumes = payload.get("globals", {}).get("volumes")
        if volumes:
            state.ws.evaluate(
                f"window.mkHeadless.loadVolumes({json.dumps(volumes)})",
                timeout_ms=timeout_ms,
            )
        # Before the scene, so the camera fit (auto or from `rotate`/`zoom`)
        # is computed against this image's aspect rather than the window's.
        state.ws.evaluate(
            f"window.mkHeadless.setViewport({width}, {height})",
            timeout_ms=timeout_ms,
        )
        if objects is not None:
            scenes = [o["scene"] for o in objects]
            globals_ = payload.get("globals", {"components": []})
        else:
            # The canvas and camera belong to the picture, so they are applied
            # once, after every object. Leaving them on an object's own scene
            # would let the next applyScene reset them: the fog is written on
            # every call, deliberately, so one render cannot inherit another's.
            scene = dict(payload["scene"])
            globals_ = {"components": []}
            for key in ("canvas", "camera"):
                if key in scene:
                    globals_[key] = scene.pop(key)
            scenes = [scene]
        state.ws.evaluate(
            f"window.mkHeadless.applyScenes({json.dumps(scenes)}, {json.dumps(globals_)})",
            timeout_ms=timeout_ms,
        )
        options = {
            "width": width,
            "height": height,
            "occlusion": QUALITY_PRESETS[quality]["occlusion"],
            "sampleLevel": QUALITY_PRESETS[quality]["sample_level"],
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
    return png
