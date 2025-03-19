from moleculekit.molecule import Molecule
from moleculekit.projections.metriccoordinate import MetricCoordinate
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

_MOL = Molecule(
    os.path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
)
_MOL.read(os.path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))


def _test_project():
    ref = _MOL.copy()
    ref.coords = np.atleast_3d(ref.coords[:, :, 0])
    metr = MetricCoordinate("protein and name CA")
    data = metr.project(_MOL)

    lastcoors = np.array(
        [
            -24.77000237,
            -27.76000023,
            -30.44000244,
            -33.65000153,
            -33.40999985,
            -36.32000351,
            -36.02000427,
            -36.38000107,
            -39.61000061,
            -41.01000214,
            -43.80000305,
            -45.56000137,
            -45.36000061,
            -47.13000488,
            -49.54000473,
            -50.6000061,
            -50.11000061,
            -52.15999985,
            -55.1400032,
            -55.73000336,
        ],
        dtype=np.float32,
    )
    assert np.all(
        np.abs(data[-1, -20:] - lastcoors) < 0.001
    ), "Coordinates calculation is broken"


def _test_project_align():
    ref = _MOL.copy()
    ref.coords = np.atleast_3d(ref.coords[:, :, 0])
    metr = MetricCoordinate("protein and name CA", ref)
    data = metr.project(_MOL)

    lastcoors = np.array(
        [
            6.79283285,
            5.55226946,
            4.49387407,
            2.94484425,
            5.36937141,
            3.18590879,
            5.75874281,
            5.48864174,
            1.69625032,
            1.58790839,
            0.57877392,
            -2.66498065,
            -3.70919156,
            -3.33702421,
            -5.38465405,
            -8.43286991,
            -8.15859032,
            -7.85062265,
            -10.92551327,
            -13.70733166,
        ],
        dtype=np.float32,
    )
    assert np.all(
        np.abs(data[-1, -20:] - lastcoors) < 0.001
    ), "Coordinates calculation is broken"


def _test_com_coordinates():
    from moleculekit.molecule import Molecule
    from moleculekit.periodictable import periodictable

    mol = Molecule().empty(4)
    mol.coords = np.array(
        [
            [0, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )[:, :, None]
    mol.element[:] = "H"
    mol.resid[:] = [0, 1, 1, 0]

    fixargs = {"pbc": False, "atomsel": "all", "centersel": "all"}

    proj = MetricCoordinate(groupsel="all", groupreduce="centroid", **fixargs).project(
        mol
    )
    assert np.array_equal(proj, [[0, 0.25, 0]])

    proj = MetricCoordinate(groupsel="all", groupreduce="com", **fixargs).project(mol)
    assert np.array_equal(proj, [[0, 0.25, 0]])

    proj = MetricCoordinate(
        groupsel="residue", groupreduce="centroid", **fixargs
    ).project(mol)
    assert np.array_equal(proj, [[0, 0, 0.5, 0, 0, 0]])

    proj = MetricCoordinate(groupsel="residue", groupreduce="com", **fixargs).project(
        mol
    )
    assert np.array_equal(proj, [[0, 0, 0.5, 0, 0, 0]])

    ref = _MOL.copy()
    ref.coords = np.atleast_3d(ref.coords[:, :, 0])

    fixargs = {
        "pbc": False,
        "atomsel": "protein and name CA",
        "centersel": "protein and name CA",
    }
    proj = MetricCoordinate(groupsel="all", groupreduce="centroid", **fixargs).project(
        _MOL
    )
    sel = _MOL.atomselect(fixargs["atomsel"])
    refproj = _MOL.coords[sel].mean(axis=0)

    assert np.array_equal(proj, refproj.T)

    proj = MetricCoordinate(groupsel="all", groupreduce="com", **fixargs).project(_MOL)
    sel = _MOL.atomselect(fixargs["atomsel"])
    masses = np.array(
        [periodictable[el].mass for el in _MOL.element[sel]], dtype=np.float32
    )
    coords = _MOL.coords[sel]

    com = coords * masses.reshape(-1, 1, 1)
    com = com.sum(axis=0) / masses.sum()
    assert np.array_equal(proj, com.T)

    proj = MetricCoordinate(groupsel="residue", groupreduce="com", **fixargs).project(
        _MOL
    )
    assert proj.shape == (200, 831)
