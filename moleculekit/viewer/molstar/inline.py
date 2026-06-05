"""Inline (server-less) Mol* viewer for Jupyter notebooks. Emits an
<iframe srcdoc> that loads Mol* from a CDN and renders the molecule entirely
from data inlined into the cell output."""

from __future__ import annotations

import base64
import html as _html
import json
import os
import struct
from typing import TYPE_CHECKING

import numpy as np

from moleculekit.util import tempname
from moleculekit.viewer.molstar.mvs import build_mvs

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


def _get_ipython():
    """Return the active IPython shell instance, or None. Isolated so tests
    can monkeypatch it."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def running_in_notebook() -> bool:
    """True only inside a Jupyter/IPython *kernel* (ZMQInteractiveShell).

    Terminal IPython and plain Python return False, so view() falls back to
    the server there.

    Returns
    -------
    in_notebook : bool
        True when running inside a Jupyter/IPython kernel, otherwise False.
    """
    shell = _get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _scene_from_reps(mol, replist) -> dict:
    """Translate a list of _Representation objects into the scene IR consumed
    by both render paths. Reps that match no atoms are dropped."""
    representations = []
    for rep in replist:
        translated = mol.reps._translateMolstar(rep)
        if translated is not None:
            representations.append(translated)
    return {"representations": representations}


MOLSTAR_CDN_VERSION = "5.9.0"
DEFAULT_HEIGHT = 420
_CDN = f"https://cdn.jsdelivr.net/npm/molstar@{MOLSTAR_CDN_VERSION}/build/viewer"

# Mirrors the docs-theme bootstrap: a clean viewport, no side panels.
_VIEWER_OPTIONS = {
    "layoutIsExpanded": False,
    "layoutShowControls": False,
    "layoutShowSequence": False,
    "layoutShowLog": False,
    "layoutShowLeftPanel": False,
    "viewportShowExpand": True,
    "viewportShowSelectionMode": False,
    "viewportShowAnimation": False,
}


def _bcif_bytes(mol) -> bytes:
    """Write the molecule to BinaryCIF and return the bytes (temp file is
    created and removed; nothing is left on disk)."""
    path = tempname(suffix=".bcif")
    try:
        mol.write(path)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def coords_to_dcd_bytes(mol: "Molecule") -> bytes:
    """Encode all frames of mol.coords (Angstrom) into a CHARMM/NAMD DCD byte
    string in memory. Lossless float32, no unit cell, no temp file.

    Parameters
    ----------
    mol : Molecule
        The molecule whose coordinate frames are encoded.

    Returns
    -------
    dcd : bytes
        The in-memory DCD-format byte string for all frames of ``mol``.
    """
    coords = mol.coords  # (natoms, 3, nframes)
    natoms = int(coords.shape[0])
    nframes = int(coords.shape[2])

    icntrl = [0] * 20
    icntrl[0] = nframes   # NSET
    icntrl[1] = 0         # ISTART
    icntrl[2] = 1         # NSAVC
    icntrl[3] = nframes   # NSTEP
    icntrl[19] = 24       # CHARMM format marker

    buf = bytearray()
    # header record (4-byte 'CORD' + 20 int32 = 84 bytes)
    buf += struct.pack("<i", 84)
    buf += b"CORD"
    buf += struct.pack("<20i", *icntrl)
    buf += struct.pack("<i", 84)
    # title record
    title = b"Created by moleculekit".ljust(80)[:80]
    rec = 4 + len(title)
    buf += struct.pack("<i", rec)
    buf += struct.pack("<i", 1)  # NTITLE
    buf += title
    buf += struct.pack("<i", rec)
    # natom record
    buf += struct.pack("<i", 4)
    buf += struct.pack("<i", natoms)
    buf += struct.pack("<i", 4)
    # coordinate records: X, then Y, then Z per frame
    block = natoms * 4
    for f in range(nframes):
        frame = np.ascontiguousarray(coords[:, :, f], dtype="<f4")
        for axis in range(3):
            buf += struct.pack("<i", block)
            buf += frame[:, axis].astype("<f4", copy=False).tobytes()
            buf += struct.pack("<i", block)
    return bytes(buf)


def _json_for_script(text: str) -> str:
    """Make a string safe to embed inside a <script type=application/json>
    block (escape the only sequence that could close the tag)."""
    return text.replace("<", "\\u003c")


class MolstarInlineView:
    """Notebook cell output that renders a Mol* viewer inline.

    Its ``_repr_html_`` returns an ``<iframe srcdoc>`` that loads Mol* from a
    CDN and renders the inlined scene/data with no running server. Construct it
    with either a MolViewSpec ``mvsj`` string (single-frame path) or a
    ``payload`` dict (multi-frame trajectory path); exactly one is used.
    """

    def __init__(self, *, height: int, mvsj: str | None = None,
                 payload: dict | None = None):
        """Create an inline viewer.

        Parameters
        ----------
        height : int
            Height of the rendered iframe in pixels.
        mvsj : str or None, optional
            A MolViewSpec scene JSON string for the single-frame path. When
            given, the viewer loads it via ``loadMvsData``.
        payload : dict or None, optional
            Trajectory payload for the multi-frame path, with base64 ``topo``
            (BinaryCIF) and ``dcd`` (DCD coordinates) entries. Used when
            ``mvsj`` is None.
        """
        self._height = height
        self._mvsj = mvsj
        self._payload = payload

    def _srcdoc(self) -> str:
        head = (
            f'<link rel="stylesheet" href="{_CDN}/molstar.css">'
            f'<script src="{_CDN}/molstar.js"></script>'
        )
        opts = json.dumps(_VIEWER_OPTIONS)
        if self._mvsj is not None:
            data_block = (
                '<script type="application/json" id="mvsj">'
                f"{_json_for_script(self._mvsj)}</script>"
            )
            init = (
                "var mvsj=document.getElementById('mvsj').textContent;"
                f"molstar.Viewer.create(document.getElementById('app'),{opts})"
                ".then(function(v){v.loadMvsData(mvsj,'mvsj');});"
            )
        else:
            data_block = (
                '<script type="application/json" id="payload">'
                f"{_json_for_script(json.dumps(self._payload))}</script>"
            )
            # Multi-frame: load the topology + DCD coordinates as a trajectory.
            # Mol* applies its default preset (cartoon/ball-and-stick + element
            # coloring) and a playback bar automatically. Custom per-selection
            # reps are not applied here (see "Known limitations").
            init = (
                "var P=JSON.parse(document.getElementById('payload').textContent);"
                "function b64(s){var b=atob(s),u=new Uint8Array(b.length);"
                "for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}"
                f"molstar.Viewer.create(document.getElementById('app'),{opts})"
                ".then(function(v){return v.loadTrajectory({"
                # Topology is BinaryCIF, but loadTrajectory's *trajectory* format
                # registry keys binary CIF under 'mmcif' (it has no 'bcif'
                # entry, unlike the structure registry the MVS path uses); its
                # mmcif provider parses binary CIF transparently.
                "model:{kind:'model-data',data:b64(P.topo),format:'mmcif'},"
                "coordinates:{kind:'coordinates-data',data:b64(P.dcd),"
                "format:'dcd',isBinary:true}}).then(function(){"
                # loadTrajectory leaves Mol*'s default canvas background; the
                # single-frame MVS path renders on white, so match it (white =
                # 0xffffff = 16777215) for a consistent look across both paths.
                "if(v.plugin.canvas3d)v.plugin.canvas3d.setProps("
                "{renderer:{backgroundColor:16777215}});});});"
            )
        body = (
            '<div id="app" style="position:absolute;inset:0"></div>'
            f"<script>{init}</script>"
        )
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"{head}</head><body style='margin:0'>{data_block}{body}</body></html>"
        )

    def _repr_html_(self) -> str:
        srcdoc = _html.escape(self._srcdoc(), quote=True)
        # sandbox: allow-scripts is required; allow-same-origin is needed by
        # Mol*'s blob web-workers in some browsers. Verify in Lab + VS Code.
        return (
            f'<iframe srcdoc="{srcdoc}" '
            f'style="width:100%;height:{self._height}px;border:none" '
            'sandbox="allow-scripts allow-same-origin"></iframe>'
        )


def build_inline_view(
    mol: "Molecule", scene: dict, *, height: int = DEFAULT_HEIGHT
) -> "MolstarInlineView":
    """Build a MolstarInlineView for ``mol``.

    A single-frame molecule takes the MVS path (custom representations from
    ``scene`` are applied); a multi-frame molecule takes the trajectory path.

    Parameters
    ----------
    mol : Molecule
        The molecule to render.
    scene : dict
        Scene description with a ``representations`` list (as produced for the
        viewer). Used only on the single-frame path.
    height : int, optional
        Height of the rendered iframe in pixels. Defaults to ``DEFAULT_HEIGHT``.

    Returns
    -------
    view : MolstarInlineView
        A notebook cell output rendering the inline viewer.
    """
    if mol.numFrames == 1:
        data_url = "data:application/octet-stream;base64," + _b64(_bcif_bytes(mol))
        rep_kwargs = {"representations": scene.get("representations") or None}
        mvsj = build_mvs(mol, structure_url=data_url, **rep_kwargs)
        return MolstarInlineView(height=height, mvsj=mvsj)
    return _build_trajectory_view(mol, scene, height)  # implemented in Phase 2


def _build_trajectory_view(mol, scene, height):
    # `scene` (custom reps) is intentionally unused on the trajectory path:
    # Mol*'s public UMD applies its default preset + playback for trajectories.
    # See "Known limitations" in the design spec.
    payload = {
        "topo": _b64(_bcif_bytes(mol)),
        "dcd": _b64(coords_to_dcd_bytes(mol)),
    }
    return MolstarInlineView(height=height, payload=payload)
