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


@pytest.mark.parametrize("span", [1e4, 1e5, 1e6, 1e7])
def _test_unwrapped_coordinates(span):
    # Two bonded atoms plus distant ones spanning a large unwrapped range must not
    # crash bond_grid_search (regression: uint32 overflow on xyz_boxes products).
    from moleculekit.bondguesser import bond_grid_search

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [span, 0.0, 0.0],
            [span, span, span],
        ],
        dtype=np.float32,
    )
    radii = np.array([1.7, 1.7, 1.7, 1.7], dtype=np.float32)
    is_h = np.zeros(4, dtype=np.uint32)

    bonds = bond_grid_search(coords, 2.04, is_h, radii)
    assert bonds.dtype == np.uint32
    assert bonds.shape == (1, 2)
    assert sorted(bonds[0].tolist()) == [0, 1]


def _test_high_atom_indices_preserved(monkeypatch):
    # Atom indices above 2**24 must survive the result-assembly path
    # (regression: float32 round-trip rounds odd indices and corrupts bonds).
    from moleculekit import bondguesser

    HIGH_A = 16777217  # 2**24 + 1: smallest int not exactly representable in float32
    HIGH_B = HIGH_A + 2  # 16777219: also misrepresented in float32 -> 16777220

    injected = {"used": False}

    def fake_grid_bonds(
        coords, radii, is_hydrogen, pairdist, boxidx, atoms_in_box, gridlist
    ):
        if injected["used"]:
            return []
        injected["used"] = True
        return [HIGH_A, HIGH_B]

    monkeypatch.setattr(bondguesser, "grid_bonds", fake_grid_bonds)

    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    radii = np.array([1.7, 1.7], dtype=np.float32)
    is_h = np.zeros(2, dtype=np.uint32)

    bonds = bondguesser.bond_grid_search(coords, 2.04, is_h, radii)
    assert bonds.dtype == np.uint32
    assert bonds.shape == (1, 2)
    assert int(bonds[0, 0]) == HIGH_A
    assert int(bonds[0, 1]) == HIGH_B


def _test_nan_coords_raises():
    from moleculekit.bondguesser import bond_grid_search

    coords = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=np.float32
    )
    radii = np.array([1.7, 1.7, 1.7], dtype=np.float32)
    is_h = np.zeros(3, dtype=np.uint32)

    with pytest.raises(ValueError, match="finite"):
        bond_grid_search(coords, 2.04, is_h, radii)


def _test_zero_grid_cutoff_raises():
    from moleculekit.bondguesser import bond_grid_search

    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    radii = np.array([1.7, 1.7], dtype=np.float32)
    is_h = np.zeros(2, dtype=np.uint32)

    with pytest.raises(ValueError, match="grid_cutoff"):
        bond_grid_search(coords, 0.0, is_h, radii)
