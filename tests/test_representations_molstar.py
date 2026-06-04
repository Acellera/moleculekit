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
