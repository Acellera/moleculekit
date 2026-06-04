import numpy as np

from moleculekit.molecule import Molecule
from moleculekit.representations import Representations
from moleculekit.viewer.molstar import inline


def _ala_mol():
    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.segid[:] = "P"
    mol.record[:] = "ATOM"
    mol.serial[:] = [1, 2, 3]
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)
    mol.coords[1, :, 0] = [1.5, 0.0, 0.0]
    mol.coords[2, :, 0] = [3.0, 0.0, 0.0]
    return mol


def test_running_in_notebook_true(monkeypatch):
    ZMQInteractiveShell = type("ZMQInteractiveShell", (), {})
    monkeypatch.setattr(inline, "_get_ipython", lambda: ZMQInteractiveShell())
    assert inline.running_in_notebook() is True


def test_running_in_notebook_false_terminal(monkeypatch):
    TerminalInteractiveShell = type("TerminalInteractiveShell", (), {})
    monkeypatch.setattr(inline, "_get_ipython", lambda: TerminalInteractiveShell())
    assert inline.running_in_notebook() is False


def test_running_in_notebook_false_no_ipython(monkeypatch):
    monkeypatch.setattr(inline, "_get_ipython", lambda: None)
    assert inline.running_in_notebook() is False


def test_scene_from_reps_maps_each_rep():
    mol = _ala_mol()
    reps = Representations(mol)
    reps.add("all", "VDW", "Name")
    scene = inline._scene_from_reps(mol, reps.replist)
    assert scene["representations"] == [
        {"atom_indices": [0, 1, 2], "type": "spacefill",
         "color": {"theme": "element-symbol"}}
    ]


def test_scene_from_reps_empty_when_no_reps():
    mol = _ala_mol()
    scene = inline._scene_from_reps(mol, [])
    assert scene["representations"] == []


def test_single_frame_repr_html_embeds_iframe_and_mvsj():
    mol = _ala_mol()  # 1 frame
    scene = inline._scene_from_reps(mol, [])
    view = inline.build_inline_view(mol, scene, height=321)
    html = view._repr_html_()
    assert "<iframe" in html
    assert "height:321px" in html or 'height="321"' in html
    assert inline.MOLSTAR_CDN_VERSION in html
    assert "loadMvsData" in html
    assert "loadTrajectory" not in html
    # the bcif is inlined as a base64 data: URL inside the mvsj
    assert "data:application/octet-stream;base64," in html


def test_single_frame_no_files_left(tmp_path, monkeypatch):
    # build_inline_view must not leave temp files in the cwd
    monkeypatch.chdir(tmp_path)
    mol = _ala_mol()
    inline.build_inline_view(mol, inline._scene_from_reps(mol, []), height=420)
    assert list(tmp_path.iterdir()) == []


def test_view_returns_inline_view_in_notebook(monkeypatch):
    mol = _ala_mol()
    monkeypatch.setattr(inline, "running_in_notebook", lambda: True)
    out = mol.view(viewer="molstar")
    assert isinstance(out, inline.MolstarInlineView)


def test_view_uses_server_outside_notebook(monkeypatch):
    mol = _ala_mol()
    monkeypatch.setattr(inline, "running_in_notebook", lambda: False)
    called = {}
    import moleculekit.viewer.molstar.server as server
    monkeypatch.setattr(server, "register", lambda m: called.setdefault("reg", m))
    out = mol.view(viewer="molstar")
    assert out is None
    assert called.get("reg") is mol


def test_dcd_roundtrip_via_moleculekit(tmp_path):
    mol = _ala_mol()
    rng = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    mol.coords = np.ascontiguousarray(rng)  # 4 frames
    data = inline.coords_to_dcd_bytes(mol)
    dcd_path = tmp_path / "traj.dcd"
    dcd_path.write_bytes(data)
    readback = mol.copy()
    readback.read(str(dcd_path))
    assert readback.numFrames == 4
    np.testing.assert_allclose(readback.coords, mol.coords, atol=1e-3)


def test_multi_frame_repr_html_uses_loadTrajectory():
    mol = _ala_mol()
    mol.coords = np.zeros((3, 3, 3), dtype=np.float32)  # 3 frames
    scene = inline._scene_from_reps(mol, [])
    view = inline.build_inline_view(mol, scene, height=300)
    html = view._repr_html_()
    src = view._srcdoc()  # raw (un-escaped) HTML; base64 has no ':' or "'"
    assert "loadTrajectory" in src
    assert "loadMvsData" not in src
    assert inline.MOLSTAR_CDN_VERSION in html
    # topology uses the trajectory registry's binary-CIF name 'mmcif'
    # (NOT 'bcif', which only exists in the structure registry)
    assert "format:'mmcif'" in src
    assert "format:'dcd'" in src
    # white canvas background to match the single-frame (MVS) view
    assert "backgroundColor:16777215" in src


def test_multi_frame_no_files_left(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mol = _ala_mol()
    mol.coords = np.zeros((3, 3, 2), dtype=np.float32)
    inline.build_inline_view(mol, inline._scene_from_reps(mol, []), height=420)
    assert list(tmp_path.iterdir()) == []
