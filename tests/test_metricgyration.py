import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_project():
    from moleculekit.molecule import Molecule
    from moleculekit.projections.metricgyration import MetricGyration
    from os import path

    mol = Molecule(
        path.join(curr_dir, "test_projections", "trajectory", "filtered.psf")
    )
    mol.read(path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))

    metr = MetricGyration("protein")
    data = metr.project(mol)

    lastrog = np.array(
        [
            18.002188,
            17.97491,
            17.973713,
            17.949549,
            17.950699,
            17.9308,
            17.928637,
            18.000408,
            18.02674,
            17.98852,
            18.015263,
            17.934515,
            17.94321,
            17.949211,
            17.93479,
            17.924484,
            17.920536,
            17.860697,
            17.849443,
            17.879776,
        ],
        dtype=np.float32,
    )
    assert np.all(
        np.abs(data[-20:, 0] - lastrog) < 0.001
    ), "Radius of gyration calculation is broken"
