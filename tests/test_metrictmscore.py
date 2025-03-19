import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_tmscore():
    from moleculekit.molecule import Molecule
    import numpy as np
    from os import path
    from moleculekit.projections.metrictmscore import MetricTMscore

    mol = Molecule(
        path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    ref = mol.copy()

    ref.dropFrames(keep=0)
    mol.dropFrames(keep=np.arange(mol.numFrames - 20, mol.numFrames))

    metr = MetricTMscore(ref, "protein and name CA")
    data = metr.project(mol)

    lasttm = np.array(
        [
            0.9633381,
            0.96441294,
            0.96553609,
            0.96088852,
            0.96288511,
            0.95677591,
            0.96544727,
            0.96359811,
            0.95658912,
            0.96893117,
            0.96623924,
            0.96064913,
            0.96207041,
            0.95947848,
            0.96657048,
            0.95993426,
            0.96543296,
            0.96806875,
            0.96437248,
            0.96144066,
        ],
        dtype=np.float32,
    )
    assert np.all(
        np.abs(data.flatten() - lasttm) < 0.001
    ), "TMscore calculation is broken"
