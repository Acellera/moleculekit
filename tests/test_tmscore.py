import numpy as np
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_tmscore():
    from moleculekit.molecule import Molecule
    from moleculekit.align import molTMscore
    import os

    expectedTMscore = np.array(
        [
            0.21418523758241995,
            0.23673770143317904,
            0.23433833284964856,
            0.21362964335736384,
            0.2093516362636725,
            0.2091586174519859,
            0.2701289508300192,
            0.2267523806405569,
            0.21230792537194731,
            0.23720109756991442,
        ]
    )
    expectedRMSD = np.array(
        [
            3.70322128077056,
            3.4363702744873135,
            3.18819300389854,
            3.844558765275783,
            3.5305388236937127,
            3.5571699112057917,
            2.9377762912738348,
            2.979786917608776,
            2.707924279670757,
            2.6305131814498712,
        ]
    )

    mol = Molecule(os.path.join(curr_dir, "test_tmscore", "filtered.pdb"))
    mol.read(os.path.join(curr_dir, "test_tmscore", "traj.xtc"))
    ref = Molecule(os.path.join(curr_dir, "test_tmscore", "ntl9_2hbb.pdb"))
    tmscore, rmsd, _ = molTMscore(
        mol, ref, "protein and name CA", "protein and name CA"
    )
    assert np.allclose(tmscore, expectedTMscore)
    assert np.allclose(rmsd, expectedRMSD)
