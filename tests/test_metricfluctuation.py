from moleculekit.projections.metricfluctuation import MetricFluctuation
from moleculekit.molecule import Molecule
import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def _ref():
    mol = Molecule(
        os.path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(os.path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    mol.dropFrames(keep=0)
    return mol


@pytest.fixture(scope="module")
def _mol():
    mol = Molecule(
        os.path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(os.path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    mol.dropFrames(
        keep=np.arange(mol.numFrames - 20, mol.numFrames)
    )  # Keep only last 20 frames
    return mol


def _test_metricfluctuation_atom_ref(_mol, _ref):
    metr = MetricFluctuation("protein and name CA", _ref)
    _ = metr.getMapping(_mol)  # Just test if it executes fine
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricfluctuation",
            "fluctuation_atom_ref.npy",
        )
    )
    assert np.allclose(
        data, refdata, atol=1e-3
    ), "Atom to ref fluctuation calculation is broken"


def _test_metricfluctuation_atom_mean(_mol):
    metr = MetricFluctuation("protein and name CA")
    _ = metr.getMapping(_mol)  # Just test if it executes fine
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricfluctuation",
            "fluctuation_atom_mean.npy",
        )
    )
    assert np.allclose(
        data, refdata, atol=1e-3
    ), "Atom mean fluctuation calculation is broken"


def _test_metricfluctuation_residue_ref(_mol, _ref):
    metr = MetricFluctuation("protein and noh", _ref, mode="residue")
    _ = metr.getMapping(_mol)  # Just test if it executes fine
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricfluctuation",
            "fluctuation_residue_ref.npy",
        )
    )
    assert np.allclose(
        data, refdata, atol=1e-3
    ), "Residue to ref fluctuation calculation is broken"


def _test_metricfluctuation_residue_mean(_mol):
    metr = MetricFluctuation("protein and noh", mode="residue")
    _ = metr.getMapping(_mol)  # Just test if it executes fine
    data = metr.project(_mol)
    refdata = np.load(
        os.path.join(
            curr_dir,
            "test_projections",
            "metricfluctuation",
            "fluctuation_residue_mean.npy",
        )
    )
    assert np.allclose(
        data, refdata, atol=1e-3
    ), "Residue mean fluctuation calculation is broken"
