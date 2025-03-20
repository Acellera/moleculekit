import numpy as np
import pytest
import sys
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from openbabel import pybel
except ImportError:
    HAS_OPENBABEL = False
else:
    HAS_OPENBABEL = True


def _test_preparation():
    from moleculekit.molecule import Molecule, mol_equal
    from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
    from os import path

    mol = Molecule(path.join(curr_dir, "test_atomtyper", "1ATL.pdb"))
    mol.remove('resname "0QI"')
    ref = Molecule(path.join(curr_dir, "test_atomtyper", "1ATL_prepared.pdb"))
    mol2 = prepareProteinForAtomtyping(mol, pH=7.0, verbose=False)

    assert mol_equal(mol2, ref, exceptFields=("coords",))


@pytest.mark.skipif(not HAS_OPENBABEL, reason="Openbabel is not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Windows OBabel fails at atom typing"
)
def _test_obabel_atomtyping():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.atomtyper import getPDBQTAtomTypesAndCharges
    from os import path

    mol = Molecule(path.join(curr_dir, "test_voxeldescriptors", "3K4X.pdb"))
    at, ch = getPDBQTAtomTypesAndCharges(mol, validitychecks=False)
    atref, chref = np.load(
        path.join(curr_dir, "test_voxeldescriptors", "3K4X_atomtypes.npy"),
        allow_pickle=True,
    )
    assert np.array_equal(at, atref)
    assert np.array_equal(ch, chref)
