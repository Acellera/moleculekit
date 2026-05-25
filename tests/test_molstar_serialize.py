import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar.serialize import molecule_to_dict


@pytest.fixture
def two_atom_mol():
    """Minimal Molecule with 2 atoms, 1 bond, 2 frames."""
    mol = Molecule().empty(2)
    mol.element[:] = ["C", "O"]
    mol.name[:] = ["C1", "O2"]
    mol.resname[:] = ["LIG", "LIG"]
    mol.resid[:] = [1, 1]
    mol.chain[:] = ["A", "A"]
    mol.segid[:] = ["L", "L"]
    mol.record[:] = ["HETATM", "HETATM"]
    mol.serial[:] = [1, 2]
    mol.formalcharge[:] = [0, -1]
    mol.bonds = np.array([[0, 1]], dtype=np.uint32)
    mol.bondtype = np.array(["2"], dtype=object)
    mol.coords = np.zeros((2, 3, 2), dtype=np.float32)
    mol.coords[0, :, 0] = [0.0, 0.0, 0.0]
    mol.coords[1, :, 0] = [1.2, 0.0, 0.0]
    mol.coords[0, :, 1] = [0.0, 0.0, 0.0]
    mol.coords[1, :, 1] = [1.5, 0.0, 0.0]
    return mol


def _test_dict_has_expected_keys(two_atom_mol):
    d = molecule_to_dict(two_atom_mol)
    expected_keys = {
        "altloc", "atomtype", "beta", "bonds", "bondtype", "chain",
        "charge", "element", "formalcharge", "insertion", "name",
        "occupancy", "record", "resid", "resname", "segid", "serial",
        "coords", "numFrames", "numAtoms",
    }
    assert set(d.keys()) == expected_keys


def _test_atom_fields_are_python_lists(two_atom_mol):
    d = molecule_to_dict(two_atom_mol)
    assert isinstance(d["element"], list)
    assert d["element"] == ["C", "O"]
    assert d["formalcharge"] == [0, -1]


def _test_bonds_are_pair_lists(two_atom_mol):
    d = molecule_to_dict(two_atom_mol)
    assert d["bonds"] == [[0, 1]]
    assert d["bondtype"] == ["2"]


def _test_default_frame_is_zero(two_atom_mol):
    d = molecule_to_dict(two_atom_mol)
    assert d["coords"] == pytest.approx([0.0, 1.2, 0.0, 0.0, 0.0, 0.0])
    assert d["numFrames"] == 2
    assert d["numAtoms"] == 2


def _test_explicit_frame_index(two_atom_mol):
    d = molecule_to_dict(two_atom_mol, frame=1)
    assert d["coords"] == pytest.approx([0.0, 1.5, 0.0, 0.0, 0.0, 0.0])


def _test_empty_arrays_serialize_clean():
    mol = Molecule().empty(1)
    mol.element[:] = ["C"]
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    d = molecule_to_dict(mol)
    assert d["bonds"] == []
    assert d["bondtype"] == []
    assert d["numAtoms"] == 1
