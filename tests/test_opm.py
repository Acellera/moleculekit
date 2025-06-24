import pytest
from moleculekit.opm import blastp, align_to_opm
from moleculekit.molecule import Molecule
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(
    not blastp,
    reason="Cannot run test without blastp and makeblastdb executables in PATH",
)
def _test_align_opm():
    mol = Molecule("7y89")
    res = align_to_opm(mol)
    assert len(res) == 3
    molaln = res[0]["hsps"][0]["aligned_mol"]
    assert molaln.coords.shape == (8641, 3, 1)


@pytest.mark.skipif(
    not blastp,
    reason="Cannot run test without blastp and makeblastdb executables in PATH",
)
def _test_align_opm_with_id():
    from moleculekit.opm import align_to_opm
    from moleculekit.molecule import Molecule

    mol = Molecule(os.path.join(curr_dir, "test_opm", "align_to_7e2y.pdb"))
    res = align_to_opm(mol, molsel="protein and chain A", opmid="7e2y", maxalignments=1)
    assert len(res) == 1
    assert len(res[0]["hsps"]) == 4
    assert res[0]["hsps"][0]["aligned_mol"].coords.shape == (3239, 3, 1)
