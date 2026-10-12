from moleculekit.opm import align_to_opm
from moleculekit.molecule import Molecule
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def test_align_opm():
    mol = Molecule("7y89")
    res = align_to_opm(mol)
    assert len(res) == 3
    molaln = res[0]["hsps"][0]["aligned_mol"]
    assert molaln.coords.shape == (8641, 3, 1)


def test_align_opm_with_id():
    from moleculekit.opm import align_to_opm
    from moleculekit.molecule import Molecule

    mol = Molecule(os.path.join(curr_dir, "test_opm", "align_to_7e2y.pdb"))
    res = align_to_opm(mol, molsel="protein and chain A", opmid="7e2y", maxalignments=1)
    assert len(res) == 1
    assert res[0]["pdbid"].lower() == "7e2y"
    # How many pairs the match splits into is the search's business. What has
    # to hold is that the best pair superposes the structure onto the entry.
    best = max(res[0]["hsps"], key=lambda h: h["TM-Score"])
    assert best["TM-Score"] > 0.95
    assert best["aligned_mol"].coords.shape == (3239, 3, 1)
