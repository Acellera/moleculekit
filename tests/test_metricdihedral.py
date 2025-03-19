import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_dihedral_traj():
    from moleculekit.molecule import Molecule
    from moleculekit.projections.metricdihedral import MetricDihedral
    from os import path

    mol = Molecule(
        path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))

    metr = MetricDihedral(protsel="protein")
    data = metr.project(mol)
    dataref = np.load(
        path.join(curr_dir, "test_projections", "metricdihedral", "ref.npy")
    )
    assert np.allclose(
        data, dataref, atol=1e-03
    ), "Diherdals calculation gave different results from reference"


def _test_dihedral_5mat():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment
    from moleculekit.projections.metricdihedral import MetricDihedral
    from os import path

    mol = Molecule("5MAT")
    mol.filter("not insertion A and not altloc A B", _logger=False)
    mol = autoSegment(mol, _logger=False)

    data = MetricDihedral().project(mol)
    dataref = np.load(
        path.join(curr_dir, "test_projections", "metricdihedral", "5mat.npy")
    )
    assert np.allclose(
        data, dataref, atol=1e-03
    ), "Diherdals calculation gave different results from reference"

    ref_idx = np.load(
        path.join(
            curr_dir,
            "test_projections",
            "metricdihedral",
            "5mat_mapping_indexes.npy",
        )
    )
    mapping = MetricDihedral().getMapping(mol)
    mapping_idx = np.vstack(mapping.atomIndexes.to_numpy())

    assert np.array_equal(mapping_idx, ref_idx), "Mapping of atom indexes has changed"


def _test_dialanine_ace_nme():
    from moleculekit.molecule import Molecule
    from moleculekit.projections.metricdihedral import MetricDihedral
    from os import path

    mol = Molecule(
        path.join(
            curr_dir,
            "test_projections",
            "metricdihedral",
            "dialanine-peptide.pdb",
        )
    )
    data = MetricDihedral().project(mol)

    refarray = np.array(
        [[-0.71247578, -0.70169669, 0.27399951, -0.96172982]], dtype=np.float32
    )
    assert np.allclose(refarray, data)
