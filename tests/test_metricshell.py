import numpy as np
from moleculekit.projections.metricshell import MetricShell
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_metricshell():
    from moleculekit.molecule import Molecule
    from os import path

    mol = Molecule(
        path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))

    metr = MetricShell(
        "protein and name CA", "resname MOL and noh", periodic="selections"
    )
    data = metr.project(mol)

    refdata = np.load(
        path.join(curr_dir, "test_projections", "metricshell", "refdata.npy")
    )

    assert np.allclose(data, refdata), "Shell density calculation is broken"


def _test_metricshell_simple():
    from moleculekit.molecule import Molecule
    import numpy as np

    mol = Molecule().empty(3)
    mol.name[:] = "CL"
    mol.resname[:] = "CL"
    mol.element[:] = "Cl"
    mol.resid[:] = np.arange(3)
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)

    mol.coords[0, :, 0] = [0, 0, 0]
    mol.coords[1, :, 0] = [0.5, 0, 0]
    mol.coords[2, :, 0] = [0, 1.5, 0]

    metr = MetricShell("name CL", "name CL", periodic=None)
    data = metr.project(mol)
    # fmt: off
    refdata = np.array([[0.01768388256576615, 0.0, 0.0, 0.0, 0.01768388256576615, 0.0, 0.0, 0.0, 0.01768388256576615, 0.0, 0.0, 0.0]])
    # fmt: on
    assert np.allclose(data, refdata)

    metr = MetricShell("name CL", "name CL", numshells=2, shellwidth=1, periodic=None)
    data = metr.project(mol)
    refdata = np.array(
        [[0.23873241, 0.03410463, 0.23873241, 0.03410463, 0.0, 0.06820926]]
    )
    assert np.allclose(data, refdata)
