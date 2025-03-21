from moleculekit.molecule import Molecule
from moleculekit.projections.metricdistance import MetricDistance, MetricSelfDistance
import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib
except ImportError:
    HAS_MATPLOTLIB = False
else:
    HAS_MATPLOTLIB = True


@pytest.fixture(scope="module")
def _mol():
    mol = Molecule(
        os.path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(os.path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    return mol


@pytest.fixture(scope="module")
def _mol_skipped():
    mol = Molecule(
        os.path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(
        os.path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"), skip=10
    )
    return mol


def _test_contacts(_mol):
    metr = MetricDistance(
        "protein and name CA",
        "resname MOL and noh",
        periodic="selections",
        metric="contacts",
        threshold=8,
    )
    data = metr.project(_mol)
    reffile = os.path.join(
        curr_dir, "test_projections", "metricdistance", "distances.npy"
    )
    refdata = np.load(reffile) < 8

    assert np.allclose(data, refdata, atol=1e-3), "Contact calculation is broken"


def _test_atomselects(_mol_skipped):
    # String atomselection
    metr = MetricSelfDistance("protein and name CA", metric="contacts", threshold=8)
    data = metr.project(_mol_skipped)
    mapping = metr.getMapping(_mol_skipped)
    assert data.shape == (20, 38226) and mapping.shape == (38226, 3)

    # Integer index atomselection
    ca_atoms = _mol_skipped.atomselect("protein and name CA", indexes=True)
    metr = MetricSelfDistance(ca_atoms, metric="contacts", threshold=8)
    data = metr.project(_mol_skipped)
    mapping = metr.getMapping(_mol_skipped)
    assert data.shape == (20, 38226) and mapping.shape == (38226, 3)

    # Boolean atomselection
    ca_atoms = _mol_skipped.atomselect("protein and name CA")
    metr = MetricSelfDistance(ca_atoms, metric="contacts", threshold=8)
    data = metr.project(_mol_skipped)
    mapping = metr.getMapping(_mol_skipped)
    assert data.shape == (20, 38226) and mapping.shape == (38226, 3)

    # Manual group atomselection with integers
    ca_atoms = _mol_skipped.atomselect("protein and name CA", indexes=True)
    metr = MetricSelfDistance(
        [ca_atoms[0::3], ca_atoms[1::3], ca_atoms[2::3]],
    )
    data = metr.project(_mol_skipped)
    mapping = metr.getMapping(_mol_skipped)
    assert data.shape == (20, 3) and mapping.shape == (3, 3)

    # Manual group atomselection with booleans
    ca_atoms = _mol_skipped.atomselect("protein and name CA", indexes=True)
    ca_bools1 = _mol_skipped.atomselect(f"index {' '.join(map(str, ca_atoms[0::3]))}")
    ca_bools2 = _mol_skipped.atomselect(f"index {' '.join(map(str, ca_atoms[1::3]))}")
    ca_bools3 = _mol_skipped.atomselect(f"index {' '.join(map(str, ca_atoms[2::3]))}")
    metr = MetricSelfDistance([ca_bools1, ca_bools2, ca_bools3])
    newdata = metr.project(_mol_skipped)
    mapping = metr.getMapping(_mol_skipped)
    assert (
        newdata.shape == (20, 3)
        and mapping.shape == (3, 3)
        and np.array_equal(data, newdata)
    )


def _test_distances_trivial():
    from moleculekit.molecule import Molecule
    import numpy as np

    mol = Molecule().empty(3)
    mol.name[:] = "C"
    mol.element[:] = "C"
    mol.chain[:] = list(
        map(str, range(3))
    )  # If they are in the same chain, no wrapping is done for distances
    mol.coords = np.zeros(
        (3, 3, 2), dtype=np.float32
    )  # Make two frames so we check if the code works for nframes
    mol.coords[1, :, 0] = [3, 3, 3]
    mol.coords[2, :, 0] = [5, 5, 5]
    mol.coords[1, :, 1] = [7, 7, 7]
    mol.coords[2, :, 1] = [6, 6, 6]

    realdistances = np.linalg.norm(mol.coords[[1, 2], :, :], axis=1).T

    metr = MetricDistance("index 0", "index 1 2", metric="distances", periodic=None)
    data = metr.project(mol)
    assert np.allclose(data, realdistances), "Trivial distance calculation is broken"

    # Test wrapped distances
    wrappedcoords = np.mod(mol.coords, 2)
    wrappedrealdistances = np.linalg.norm(wrappedcoords[[1, 2], :, :], axis=1).T

    mol.box = np.full((3, 2), 2, dtype=np.float32)  # Make box 2x2x2A large
    metr = MetricDistance(
        "index 0", "index 1 2", metric="distances", periodic="selections"
    )
    data = metr.project(mol)
    assert np.allclose(
        data, wrappedrealdistances
    ), "Trivial wrapped distance calculation is broken"

    # Test min distances
    metr = MetricDistance(
        "index 0",
        "index 1 2",
        metric="distances",
        periodic=None,
        groupsel1="all",
        groupsel2="all",
    )
    data = metr.project(mol)
    assert np.allclose(
        data.flatten(), np.min(realdistances, axis=1)
    ), "Trivial distance calculation is broken"

    # Test ordering
    mol = Molecule().empty(4)
    mol.name[:] = "C"
    mol.element[:] = "C"
    mol.chain[:] = list(
        map(str, range(4))
    )  # If they are in the same chain, no wrapping is done for distances
    mol.coords = np.zeros(
        (4, 3, 2), dtype=np.float32
    )  # Make two frames so we check if the code works for nframes
    mol.coords[1, :, 0] = [1, 1, 1]
    mol.coords[2, :, 0] = [3, 3, 3]
    mol.coords[3, :, 0] = [5, 5, 5]
    mol.coords[1, :, 1] = [1, 1, 1]
    mol.coords[2, :, 1] = [7, 7, 7]
    mol.coords[3, :, 1] = [6, 6, 6]

    realdistances = np.linalg.norm(mol.coords[[2, 3], :, :] - mol.coords[0], axis=1).T
    realdistances = np.hstack(
        (
            realdistances,
            np.linalg.norm(mol.coords[[2, 3], :, :] - mol.coords[1], axis=1).T,
        )
    )

    metr = MetricDistance("index 0 1", "index 2 3", metric="distances", periodic=None)
    data = metr.project(mol)
    assert np.allclose(
        data, realdistances
    ), "Trivial distance calculation has broken ordering"


def _test_distances(_mol):
    metr = MetricDistance(
        "protein and name CA",
        "resname MOL and noh",
        metric="distances",
        periodic="selections",
    )
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(curr_dir, "test_projections", "metricdistance", "distances.npy")
    )
    assert np.allclose(data, refdata, atol=1e-3), "Distance calculation is broken"


def _test_mindistances(_mol):
    metr = MetricDistance(
        "protein and noh",
        "resname MOL and noh",
        periodic="selections",
        groupsel1="residue",
        groupsel2="all",
    )
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(curr_dir, "test_projections", "metricdistance", "mindistances.npy")
    )
    assert np.allclose(
        data, refdata, atol=1e-3
    ), "Minimum distance calculation is broken"


def _test_mindistances_truncate(_mol):
    metr = MetricDistance(
        "protein and noh",
        "resname MOL and noh",
        periodic="selections",
        groupsel1="residue",
        groupsel2="all",
        truncate=3,
    )
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(curr_dir, "test_projections", "metricdistance", "mindistances.npy")
    )
    assert np.allclose(
        data, np.clip(refdata, 0, 3), atol=1e-3
    ), "Minimum distance calculation is broken"


def _test_selfmindistance_manual(_mol):
    metr = MetricDistance(
        "protein and resid 1 to 50 and noh",
        "protein and resid 1 to 50 and noh",
        periodic=None,
        groupsel1="residue",
        groupsel2="residue",
    )
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricdistance",
            "selfmindistance.npy",
        )
    )
    assert np.allclose(data, refdata, atol=1e-3), "Manual self-distance is broken"


def _test_selfmindistance_auto(_mol):
    metr = MetricSelfDistance("protein and resid 1 to 50 and noh", groupsel="residue")
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricdistance",
            "selfmindistance.npy",
        )
    )
    assert np.allclose(data, refdata, atol=1e-3), "Automatic self-distance is broken"


def _test_mindistances_skip(_mol_skipped):
    metr = MetricSelfDistance("protein and resid 1 to 50 and noh", groupsel="residue")
    data = metr.project(_mol_skipped)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricdistance",
            "selfmindistance.npy",
        )
    )
    assert np.allclose(
        data, refdata[::10, :], atol=1e-3
    ), "Minimum distance calculation with skipping is broken"


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib is not installed")
def _test_reconstruct_contact_map():
    from moleculekit.util import tempname
    from moleculekit.molecule import Molecule
    from moleculekit.projections.metricdistance import (
        contactVecToMatrix,
        reconstructContactMap,
    )
    import matplotlib

    matplotlib.use("Agg")

    mol = Molecule("1yu8")

    metr = MetricSelfDistance("protein and name CA", metric="contacts")
    data = metr.project(mol)[0]
    mapping = metr.getMapping(mol)

    tmpfile = tempname(suffix=".svg")
    cm, newmapping, uqAtomGroups = contactVecToMatrix(data, mapping.atomIndexes)
    reconstructContactMap(data, mapping, outfile=tmpfile)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricdistance",
            "contactvectomatrix.npz",
        )
    )

    assert np.array_equal(refdata["cm"], cm)
    assert np.array_equal(refdata["newmapping"], newmapping)
    assert np.array_equal(refdata["uqAtomGroups"], uqAtomGroups)


def _test_description(_mol):
    metr = MetricDistance(
        "protein and noh", "resname MOL and noh", truncate=3, periodic="selections"
    )
    # data = metr.project(_mol)
    atomIndexes = metr.getMapping(_mol).atomIndexes.values
    refdata = np.load(
        os.path.join(curr_dir, "test_projections", "metricdistance", "description.npy"),
        allow_pickle=True,
    )
    assert np.array_equal(refdata, atomIndexes)


def _test_periodicity(_mol):
    metr = MetricDistance(
        "protein and resid 1 to 20 and noh",
        "resname MOL and noh",
        periodic="selections",
    )
    data1 = metr.project(_mol)
    metr = MetricDistance(
        "protein and resid 1 to 20 and noh",
        "resname MOL and noh",
        periodic="chains",
    )
    data2 = metr.project(_mol)
    assert np.allclose(data1, data2)

    mol_tmp = _mol.copy()
    mol_tmp.chain[:] = ""
    metr = MetricDistance(
        "protein and resid 1 to 20 and noh",
        "resname MOL and noh",
        periodic="chains",
    )
    data3 = metr.project(mol_tmp)
    assert not np.allclose(data1, data3)


def _test_com_distances():
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

    fixargs = {
        "periodic": None,
        "groupsel1": "all",
        "groupsel2": "all",
        "groupreduce1": "com",
        "groupreduce2": "com",
    }

    dist = MetricDistance("index 0", "index 1", **fixargs).project(mol)
    assert np.abs(dist[0][0] - 1) < 1e-5

    dist = MetricDistance("index 0 1", "index 2", **fixargs).project(mol)
    assert np.abs(dist[0][0] - 1.5) < 1e-5

    dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
    assert np.abs(dist[0][0] - 1) < 1e-5

    fixargs["groupsel1"] = "residue"
    dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
    assert np.allclose(dist, [[1, 1]])

    fixargs["groupsel1"] = "all"
    fixargs["groupreduce1"] = "closest"
    dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
    assert np.allclose(dist, [[1]])

    fixargs["groupsel1"] = "residue"
    fixargs["groupreduce1"] = "closest"
    dist = MetricDistance("index 0 1 2", "index 3", **fixargs).project(mol)
    assert np.allclose(dist, [[1, 1.4142135]])

    # Real test
    mol = Molecule("3ptb")

    sel1 = "protein"
    sel2 = "resname BEN"

    mol1 = mol.copy()
    mol1.filter(sel1)
    masses = np.array([periodictable[el].mass for el in mol1.element], dtype=np.float32)
    com1 = np.sum(mol1.coords[:, :, 0] * masses[:, None], axis=0) / masses.sum()

    mol2 = mol.copy()
    mol2.filter(sel2)
    masses = np.array([periodictable[el].mass for el in mol2.element], dtype=np.float32)
    com2 = np.sum(mol2.coords[:, :, 0] * masses[:, None], axis=0) / masses.sum()

    dist = MetricDistance(
        sel1,
        sel2,
        None,
        groupsel1="all",
        groupsel2="all",
        groupreduce1="com",
        groupreduce2="com",
    ).project(mol)

    assert (
        np.abs(np.linalg.norm(com1 - com2) - dist) < 1e-2
    ), f"{np.linalg.norm(com1 - com2)}, {dist}"

    dist = MetricDistance(
        sel1,
        sel2,
        None,
        groupsel1="all",
        groupsel2="all",
        groupreduce1="com",
        groupreduce2="closest",
    ).project(mol)
    assert np.abs(dist[0][0] - 8.978174) < 1e-5

    dist = MetricDistance(
        sel1,
        sel2,
        None,
        groupsel1="all",
        groupsel2="all",
        groupreduce1="closest",
        groupreduce2="com",
    ).project(mol)
    assert np.abs(dist[0][0] - 3.8286476) < 1e-5

    dist = MetricDistance(
        sel1,
        sel2,
        None,
        groupsel1="all",
        groupsel2="all",
        groupreduce1="closest",
        groupreduce2="closest",
    ).project(mol)
    assert np.abs(dist[0][0] - 2.8153415) < 1e-5


def _test_pair_distances():
    from moleculekit.molecule import Molecule
    from moleculekit.projections.metricdistance import MetricDistance

    sel1 = np.array([0, 1, 2]).reshape(-1, 1)
    sel2 = np.array([1, 2, 3]).reshape(-1, 1)
    mol = Molecule("3ptb")
    res = MetricDistance(sel1, sel2, None, pairs=True).project(mol)
    ref = np.linalg.norm(
        mol.coords[sel1.flatten(), :, 0] - mol.coords[sel2.flatten(), :, 0], axis=1
    )
    assert np.allclose(res, ref)
    mmp = MetricDistance(sel1, sel2, None, pairs=True).getMapping(mol)
    assert mmp.shape == (3, 3)

    res = MetricDistance(
        "residue 1 2",
        "residue 3 4",
        None,
        pairs=True,
        groupsel1="residue",
        groupsel2="residue",
    ).project(mol)
    res1 = MetricDistance("residue 1", "residue 3", None).project(mol)
    res2 = MetricDistance("residue 2", "residue 4", None).project(mol)

    assert np.allclose(res, [[res1.min(), res2.min()]])
