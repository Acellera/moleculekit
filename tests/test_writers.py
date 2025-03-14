import pytest
import os


@pytest.mark.parametrize("filetype", ["xtc", "dcd", "trr"])
def _test_trajectory_writers_roundtrip(tmp_path, filetype):
    from moleculekit.molecule import Molecule
    import numpy as np

    natoms = 7
    nframes = 13

    mol = Molecule().empty(natoms)
    mol.coords = np.random.rand(natoms, 3, nframes).astype(np.float32) * 10
    mol.box = np.random.rand(3, nframes).astype(np.float32) * 10

    # The unitcell conversions fail if the angles are not realistic (not sure the exact conditions)
    mol.boxangles = np.array([[120, 120, 90]], dtype=np.float32).T
    mol.boxangles = np.tile(mol.boxangles, (1, nframes))

    mol.time = np.arange(nframes).astype(np.float32)
    mol.step = np.arange(nframes).astype(np.int32)

    mol.write(os.path.join(tmp_path, f"output.{filetype}"))

    mol2 = Molecule(os.path.join(tmp_path, f"output.{filetype}"))
    assert np.allclose(mol.coords, mol2.coords)
    assert np.allclose(mol.box, mol2.box)
    assert np.allclose(mol.boxangles, mol2.boxangles)
