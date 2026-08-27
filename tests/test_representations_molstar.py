import numpy as np

from moleculekit.molecule import Molecule
from moleculekit.representations import _Representation


def _mol():
    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)
    return mol


def test_translate_style_and_theme_color():
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("all", "NewCartoon", "Name"))
    assert out["type"] == "cartoon"
    assert out["color"] == {"theme": "element-symbol"}
    assert out["atom_indices"] == [0, 1, 2]


def test_translate_vdw_and_secondary_structure():
    mol = _mol()
    out = mol.reps._translateMolstar(
        _Representation("all", "VDW", "Secondary Structure")
    )
    assert out["type"] == "spacefill"
    assert out["color"] == {"theme": "secondary-structure"}


def test_translate_colorid_int_becomes_uniform_hex():
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("all", "Licorice", 1))
    assert out["type"] == "ball_and_stick"
    assert out["color"] == "#ff0000"


def test_translate_unmatched_selection_returns_none():
    mol = _mol()
    out = mol.reps._translateMolstar(_Representation("resname ZZZ", "Lines", "Name"))
    assert out is None


def _trypsin():
    from pathlib import Path

    return Molecule(str(Path(__file__).parent / "test_molecule" / "3ptb_filtered.pdb"))


def test_add_defaults_lists_the_automatic_scene():
    from moleculekit.viewer.molstar.scene import build_scene

    mol = _trypsin()
    mol.reps.addDefaults()
    assert [r.sel for r in mol.reps.replist] == [
        "protein or nucleic",
        "not (protein or nucleic)",
    ]

    # Every atom is drawn, and none of them twice.
    covered = sum(mol.atomselect(r.sel).astype(int) for r in mol.reps.replist)
    assert (covered == 1).all()

    scene = build_scene(mol, mol.reps.replist)
    types = [c["representation"]["type"] for c in scene["components"]]
    assert types == ["cartoon", "ball_and_stick"]


def test_add_defaults_on_a_small_molecule_is_a_single_representation():
    mol = _mol()
    mol.reps.addDefaults()
    assert [(r.sel, r.style) for r in mol.reps.replist] == [("all", "CPK")]


def test_add_defaults_omits_a_hetero_representation_matching_nothing():
    mol = _trypsin()
    mol.filter("protein")
    mol.reps.addDefaults()
    assert [r.sel for r in mol.reps.replist] == ["protein or nucleic"]
