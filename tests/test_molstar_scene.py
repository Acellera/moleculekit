import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar.scene import build_scene


def _protein():
    """Eight glycines: above MIN_CARTOON_RESIDUES, so the cartoon rule fires."""
    n = 8
    mol = Molecule().empty(n)
    mol.element[:] = "C"
    mol.name[:] = "CA"
    mol.resname[:] = "GLY"
    mol.resid[:] = np.arange(1, n + 1)
    mol.chain[:] = "A"
    mol.segid[:] = "P"
    mol.record[:] = "ATOM"
    mol.serial[:] = np.arange(1, n + 1)
    mol.coords = np.zeros((n, 3, 1), dtype=np.float32)
    mol.coords[:, 0, 0] = np.arange(n, dtype=np.float32) * 3.8
    return mol


def _ligand():
    """Two hetero atoms: below MIN_CARTOON_RESIDUES, ball-and-stick everything."""
    mol = Molecule().empty(2)
    mol.element[:] = ["C", "O"]
    mol.name[:] = ["C1", "O2"]
    mol.resname[:] = "LIG"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.segid[:] = "L"
    mol.record[:] = "HETATM"
    mol.serial[:] = [1, 2]
    mol.formalcharge[:] = [0, -1]
    mol.coords = np.zeros((2, 3, 1), dtype=np.float32)
    mol.coords[1, :, 0] = [1.2, 0.0, 0.0]
    return mol


def _selectors(scene):
    return [c["select"] for c in scene["components"]]


def test_polymer_gets_a_secondary_structure_cartoon():
    scene = build_scene(_protein())
    cartoon = [
        c for c in scene["components"] if c["representation"]["type"] == "cartoon"
    ]
    assert len(cartoon) == 1
    assert cartoon[0]["select"] == {"kind": "builtin", "name": "polymer"}
    assert cartoon[0]["color"] == {"theme": "secondary-structure"}


def test_small_molecule_gets_ball_and_stick_everything():
    scene = build_scene(_ligand())
    assert _selectors(scene) == [{"kind": "builtin", "name": "all"}]
    assert scene["components"][0]["representation"]["type"] == "ball_and_stick"


def test_reps_replace_the_automatic_scene():
    """VMD convention: setting mol.reps means those reps ARE the scene."""
    mol = _protein()
    mol.reps.add("resname GLY", style="VDW", color="red")
    scene = build_scene(mol, mol.reps.replist)

    assert len(scene["components"]) == 1
    only = scene["components"][0]
    assert only["select"]["kind"] == "atoms"
    assert only["representation"]["type"] == "spacefill"
    assert only["color"] == {"uniform": "red"}
    assert not any(
        c["representation"]["type"] == "cartoon" for c in scene["components"]
    )


def test_empty_reps_keeps_the_automatic_scene():
    mol = _protein()
    scene = build_scene(mol, [])
    assert any(c["representation"]["type"] == "cartoon" for c in scene["components"])


def test_a_rep_matching_no_atoms_warns_and_is_dropped(caplog):
    mol = _protein()
    mol.reps.add("resname GLY", style="VDW", color="red")
    mol.reps.add("resname NOPE", style="VDW", color="blue")
    with caplog.at_level("WARNING"):
        scene = build_scene(mol, mol.reps.replist)
    assert len(scene["components"]) == 1
    assert "resname NOPE" in caplog.text


def test_all_reps_matching_nothing_raises():
    """With replace semantics this would otherwise render a blank image."""
    mol = _protein()
    mol.reps.add("resname NOPE", style="VDW", color="blue")
    with pytest.raises(ValueError, match="matched no atoms"):
        build_scene(mol, mol.reps.replist)


def test_formal_charge_labels_use_mol_frame():
    mol = _ligand()
    mol.coords = np.zeros((2, 3, 2), dtype=np.float32)
    mol.coords[:, :, 1] = [[0.0, 0.0, 70.0], [1.0, 0.0, 70.0]]
    mol.frame = 1
    scene = build_scene(mol)
    assert [lab["position"] for lab in scene["labels"]] == [[1.0, 0.0, 70.0]]
    assert [lab["atom"] for lab in scene["labels"]] == [1]
    assert scene["labels"][0]["text"] == "-1"


def test_camera_carries_rotation_and_inverted_zoom():
    scene = build_scene(_protein(), rotate="top", zoom=2.0)
    assert scene["camera"]["direction"] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    assert scene["camera"]["radius_factor"] == pytest.approx(0.5)


def test_no_camera_key_without_camera_arguments():
    assert "camera" not in build_scene(_protein())


def test_background_lands_on_the_canvas():
    scene = build_scene(_protein(), background_color="white")
    assert scene["canvas"] == {"background": "white"}


def test_highlight_bonds_become_tubes():
    mol = _ligand()
    scene = build_scene(mol, highlight_bonds=[("name C1", "name O2")])
    assert scene["tubes"][0]["start"] == [0.0, 0.0, 0.0]
    assert scene["tubes"][0]["end"] == pytest.approx([1.2, 0.0, 0.0])
    assert scene["tubes"][0]["color"] == "orange"


def test_viewer_and_renderer_get_the_same_description():
    """The property this design exists for: one molecule, one scene, whichever
    path asks for it.

    ``renderer_scene`` is obtained by calling ``render.py``'s own
    ``_scene_description`` helper (the same one ``render()`` calls), not by
    re-typing its arguments here, so a drift in what ``render()`` actually
    passes to ``build_scene`` cannot sail past this test unnoticed.
    """
    from moleculekit.viewer.molstar.render import _scene_description
    from moleculekit.viewer.molstar.server import _topology_event
    from moleculekit.viewer.molstar.registry import Registry

    mol = _protein()
    mol.reps.add("resid 1 to 4", style="VDW", color="red")
    mol._tempreps.add("resid 5 to 8", style="Licorice", color="blue")

    renderer_scene = _scene_description(mol)
    registry = Registry()
    uid = registry.register(mol)
    viewer_scene = _topology_event(registry.slots[uid])["scene"]

    assert viewer_scene == renderer_scene
