from moleculekit.molecule import Molecule
from moleculekit.tools.sybyl import sybylTypes
import os
import pytest

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def mixed_templating():
    """1A4W: sulfotyrosine gets a template, the argatroban-like QWE does not.

    The realistic case, since a caller templates the residues it has SMILES
    for and leaves the rest. Bond orders then exist in the molecule without
    existing on every residue in it.
    """
    mol = Molecule(
        os.path.join(
            curr_dir, "test_systemprepare", "test-nonstandard-residues", "1A4W", "1A4W.pdb"
        )
    )
    mol.remove("element H", _logger=False)
    mol.templateResidueFromSmiles(
        "resname TYS",
        "c1cc(ccc1C[C@@H](C(=O)O)N)OS(=O)(=O)[O-]",
        addHs=True,
        _logger=False,
    )
    return mol


def test_templated_residue_is_typed(mixed_templating):
    types = sybylTypes(mixed_templating, "resname TYS")
    assert types, "the templated residue states bond orders, so it must be typed"
    # The aryl sulfate: a sulfur with two S=O, and the ring perceived aromatic.
    assert "S.o2" in types.values()
    assert "C.ar" in types.values()


def test_untemplated_residue_is_left_alone(mixed_templating):
    """A residue with no bond orders of its own must not be typed from guesses.

    Bond orders elsewhere in the molecule say nothing about this residue, and
    typing it from guessed single bonds is worse than the geometric perception
    it would replace: the quinoline ring reads as sp3, the sulfonyl sulfur as
    S.3, and the guanidinium carbon loses the C.2 that PROPKA needs to find
    the charged group.
    """
    types = sybylTypes(mixed_templating, "resname QWE")
    assert types == {}, f"untemplated QWE was typed from guessed bonds: {types}"


def test_phosphorus_typed_without_bond_orders():
    """A phosphate keeps its P=O even when nothing records a bond order.

    The only judgement here that needs no orders: a P(V) oxo species has
    exactly one P=O however the input drew it, so the remaining terminal
    oxygens are the ionizable ones. PROPKA's own 1.3 A double-bond threshold is
    calibrated for carbon and never fires at the ~1.5 A of a P=O, which made
    every terminal oxygen an independent site.
    """
    mol = Molecule(os.path.join(curr_dir, "test_systemprepare", "3U5S", "3U5S.pdb"))
    mol.remove("element H", _logger=False)
    fad = mol.atomselect("resname FAD")
    types = sybylTypes(mol, fad)

    assert types, "the FAD phosphates must be typed with or without orders"
    assert set(types.values()) <= {"P.3", "O.2"}, (
        f"only phosphorus and its P=O may be typed without orders: {types}"
    )
    # Two phosphorus atoms in the diphosphate, one P=O each.
    assert sorted(types.values()) == ["O.2", "O.2", "P.3", "P.3"]
