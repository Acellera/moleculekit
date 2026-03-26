import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "pdbid",
    [
        "3ptb",
        "3hyd",
        "6a5j",
        "5vbl",
        "7q5b",
        "1unc",
        "3zhi",
        "1a25",
        "1u5u",
        "1gzm",
        "6va1",
        "1bna",
        "3wbm",
        "1awf",
        "5vav",
    ],
)
def _test_bond_guessing(pdbid):
    from moleculekit.molecule import Molecule, calculateUniqueBonds
    from moleculekit.bondguesser import guess_bonds
    import os

    mol = Molecule(pdbid)
    bonds = guess_bonds(mol)
    bonds, _ = calculateUniqueBonds(bonds.astype(np.uint32), [])

    reff = os.path.join(curr_dir, "test_bondguesser", f"{pdbid}.csv")
    bondsref = np.loadtxt(reff, delimiter=",").astype(np.uint32)

    # with open(reff, "w") as f:
    #     for b in range(bondsref.shape[0]):
    #         f.write(f"{bondsref[b, 0]},{bondsref[b, 1]}\n")

    assert np.array_equal(bonds, bondsref)


def _test_zero_atoms():
    from moleculekit.molecule import Molecule
    from moleculekit.bondguesser import guess_bonds

    mol = Molecule()
    bonds = guess_bonds(mol)
    assert bonds.shape == (0, 2)
    assert bonds.dtype == np.uint32


def _test_single_atom():
    from moleculekit.molecule import Molecule
    from moleculekit.bondguesser import guess_bonds

    mol = Molecule()
    mol.empty(1)
    mol.element[:] = "C"
    mol.name[:] = "CA"
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)

    bonds = guess_bonds(mol)
    assert bonds.shape == (0, 2)
    assert bonds.dtype == np.uint32


def _test_solvated_bond_guessing():
    from moleculekit.molecule import Molecule, calculateUniqueBonds
    from moleculekit.bondguesser import guess_bonds
    import os

    mol = Molecule(os.path.join(curr_dir, "test_bondguesser", "3ptb_solvated.pdb"))
    bonds = guess_bonds(mol)
    bonds, _ = calculateUniqueBonds(bonds.astype(np.uint32), [])

    reff = os.path.join(curr_dir, "test_bondguesser", "3ptb_solvated.csv")
    bondsref = np.loadtxt(reff, delimiter=",").astype(np.uint32)

    assert np.array_equal(bonds, bondsref)
