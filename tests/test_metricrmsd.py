import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_metricrmsd():
    from moleculekit.molecule import Molecule
    import numpy as np
    from os import path
    from moleculekit.projections.metricrmsd import MetricRmsd

    mol = Molecule(
        path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    ref = mol.copy()

    ref.dropFrames(keep=0)
    mol.dropFrames(
        keep=np.arange(mol.numFrames - 20, mol.numFrames)
    )  # Keep only last 20 frames

    metr = MetricRmsd(ref, "protein and name CA")
    data = metr.project(mol)

    lastrmsd = np.array(
        [
            1.30797791,
            1.29860222,
            1.25042927,
            1.31319737,
            1.27044261,
            1.40294552,
            1.25354612,
            1.30127883,
            1.40618336,
            1.18303752,
            1.24414587,
            1.34513164,
            1.31932807,
            1.34282494,
            1.2261436,
            1.36359048,
            1.26243281,
            1.21157813,
            1.26476419,
            1.29413617,
        ],
        dtype=np.float32,
    )
    assert np.all(np.abs(data[-20:] - lastrmsd) < 0.001), "RMSD calculation is broken"
