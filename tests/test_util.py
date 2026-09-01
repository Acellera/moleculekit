from moleculekit.molecule import Molecule
import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

_MOLDIALA = Molecule(os.path.join(curr_dir, "pdb", "alanine.pdb"))


def test_guessAnglesDihedrals():
    from moleculekit.util import calculateAnglesAndDihedrals

    mol = Molecule(os.path.join(curr_dir, "pdb", "NH4.pdb"))
    angles, dihedrals = calculateAnglesAndDihedrals(mol.bonds)

    assert angles.dtype == np.uint32, "Returned wrong dtype for angles"
    assert dihedrals.dtype == np.uint32, "Returned wrong dtype for dihedrals"
    assert np.all(angles.shape == (6, 3)), "Returned wrong number of angles"
    assert np.all(dihedrals.shape == (0, 4)), "Returned wrong number of dihedrals"


def test_mol_rmsd():
    from moleculekit.util import molRMSD, rotationMatrix

    mol = _MOLDIALA
    mol2 = mol.copy()
    mol2.rotateBy(rotationMatrix([1, 0, 0], np.pi / 3))
    rmsd = molRMSD(mol, mol2, np.arange(mol.numAtoms), np.arange(mol2.numAtoms))

    assert np.allclose(rmsd, 5.4344)


def test_orientOnAxes():
    from moleculekit.util import orientOnAxes

    omol = orientOnAxes(_MOLDIALA)

    covariance = np.cov(omol.coords[:, :, 0].T)
    _, eigenvectors = np.linalg.eigh(covariance)

    assert np.allclose(np.diag(eigenvectors), np.array([1, 1, 1]))
    assert (
        eigenvectors[~np.eye(eigenvectors.shape[0], dtype=bool)].max() < 1e-8
    )  # off diagonals close to 0


def test_missingChain():
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


def test_missingSegid():
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


def test_maxDistance():
    from moleculekit.util import maxDistance

    dist = maxDistance(_MOLDIALA)
    assert np.allclose(dist, 10.771703745561421)


def test_renamed_arguments_accepts_the_old_spelling():
    """Renaming a public argument must not break callers using the old name.

    The point of the decorator is that the signature carries only the new
    name, so the docs, IDEs and docstring linting see one API, while the old
    name keeps working until it is dropped.
    """
    import inspect
    import warnings

    from moleculekit.util import renamed_arguments

    @renamed_arguments(guessBonds="guess_bonds")
    def f(sel, guess_bonds=True):
        return sel, guess_bonds

    assert list(inspect.signature(f).parameters) == ["sel", "guess_bonds"]
    assert f("protein", guess_bonds=False) == ("protein", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f("protein", guessBonds=False) == ("protein", False)
    # FutureWarning, not DeprecationWarning: the latter is hidden by default
    # outside __main__, so a library's users never see it.
    assert caught[0].category is FutureWarning
    assert "guessBonds" in str(caught[0].message)


def test_renamed_arguments_refuses_both_spellings_at_once():
    import pytest

    from moleculekit.util import renamed_arguments

    @renamed_arguments(guessBonds="guess_bonds")
    def f(guess_bonds=True):
        return guess_bonds

    with pytest.raises(TypeError, match="got both 'guessBonds' and its new name"):
        f(guessBonds=False, guess_bonds=True)
