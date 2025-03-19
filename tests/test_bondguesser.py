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
