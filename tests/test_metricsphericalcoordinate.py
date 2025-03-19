import numpy as np
from moleculekit.projections.metricsphericalcoordinate import MetricSphericalCoordinate
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_metricsphericalcoordinate():
    from moleculekit.molecule import Molecule
    from os import path

    mol = Molecule(
        path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    ref = mol.copy()
    mol.read(path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    mol.bonds = mol._guessBonds()

    res = MetricSphericalCoordinate(ref, "resname MOL", "within 8 of resid 98").project(
        mol
    )
    _ = MetricSphericalCoordinate(
        ref, "resname MOL", "within 8 of resid 98"
    ).getMapping(mol)

    ref_array = np.load(
        path.join(curr_dir, "test_projections", "metricsphericalcoordinate", "res.npy")
    )
    assert np.allclose(res, ref_array, rtol=0, atol=1e-04)
