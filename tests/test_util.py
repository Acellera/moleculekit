from moleculekit.molecule import Molecule
import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

_MOLDIALA = Molecule(os.path.join(curr_dir, "pdb", "alanine.pdb"))


def _test_guessAnglesDihedrals():
    from moleculekit.util import calculateAnglesAndDihedrals

    mol = Molecule(os.path.join(curr_dir, "pdb", "NH4.pdb"))
    angles, dihedrals = calculateAnglesAndDihedrals(mol.bonds)

    assert angles.dtype == np.uint32, "Returned wrong dtype for angles"
    assert dihedrals.dtype == np.uint32, "Returned wrong dtype for dihedrals"
    assert np.all(angles.shape == (6, 3)), "Returned wrong number of angles"
    assert np.all(dihedrals.shape == (0, 4)), "Returned wrong number of dihedrals"


def _test_mol_rmsd():
    from moleculekit.util import molRMSD, rotationMatrix

    mol = _MOLDIALA
    mol2 = mol.copy()
    mol2.rotateBy(rotationMatrix([1, 0, 0], np.pi / 3))
    rmsd = molRMSD(mol, mol2, np.arange(mol.numAtoms), np.arange(mol2.numAtoms))

    assert np.allclose(rmsd, 5.4344)


def _test_orientOnAxes():
    from moleculekit.util import orientOnAxes

    omol = orientOnAxes(_MOLDIALA)

    covariance = np.cov(omol.coords[:, :, 0].T)
    _, eigenvectors = np.linalg.eigh(covariance)

    assert np.allclose(np.diag(eigenvectors), np.array([1, 1, 1]))
    assert (
        eigenvectors[~np.eye(eigenvectors.shape[0], dtype=bool)].max() < 1e-8
    )  # off diagonals close to 0


def _test_missingChain():
    from moleculekit.util import _missingChain

    mol = _MOLDIALA.copy()

    with pytest.raises(RuntimeError):
        _missingChain(mol)

    mol.chain[:] = "A"
    try:
        _missingChain(mol)
    except RuntimeError:
        raise RuntimeError("_missingChain() raised RuntimeError unexpectedly!")

    mol.chain[6] = ""
    with pytest.raises(RuntimeError):
        _missingChain(mol)


def _test_missingSegid():
    from moleculekit.util import _missingSegID

    mol = _MOLDIALA.copy()

    mol.segid[:] = ""
    with pytest.raises(RuntimeError):
        _missingSegID(mol)

    mol.segid[:] = "A"
    try:
        _missingSegID(mol)
    except RuntimeError:
        raise RuntimeError("_missingSegID() raised RuntimeError unexpectedly!")

    mol.segid[6] = ""
    with pytest.raises(RuntimeError):
        _missingSegID(mol)


def _test_maxDistance():
    from moleculekit.util import maxDistance

    dist = maxDistance(_MOLDIALA)
    assert np.allclose(dist, 10.771703745561421)
