import pytest
from moleculekit.opm import blastp, align_to_opm
from moleculekit.molecule import Molecule


@pytest.skipIf(
    not blastp, "Cannot run test without blastp and makeblastdb executables in PATH"
)
def _test_align_opm():
    mol = Molecule("7y89")
    res = align_to_opm(mol)
    assert len(res) == 3
    assert res[0]["pdbid"] == "6DDE"
    assert res[0]["thickness"] == 31.4
    molaln = res[0]["hsps"][0]["aligned_mol"]
    assert molaln.numAtoms == 8641
