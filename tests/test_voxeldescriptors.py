import numpy as np
import pytest
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from openbabel import pybel
except ImportError:
    HAS_OPENBABEL = False
else:
    HAS_OPENBABEL = True


@pytest.fixture(scope="module")
def _mol():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(os.path.join(curr_dir, "test_voxeldescriptors", "3PTB.pdb"))
    mol.filter("protein")
    mol = autoSegment(mol, field="both")
    mol = systemPrepare(mol, pH=7.0)
    mol.bonds = mol._guessBonds()
    return mol


def _test_radii(_mol):
    from moleculekit.tools.voxeldescriptors import _getChannelRadii

    sigmas = _getChannelRadii(_mol.element)
    refsigmas = np.load(
        os.path.join(curr_dir, "test_voxeldescriptors", "3PTB_sigmas.npy"),
        allow_pickle=True,
    )
    assert np.allclose(sigmas, refsigmas)


def _test_celecoxib():
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

    sm = SmallMol(os.path.join(curr_dir, "test_voxeldescriptors", "celecoxib.mol2"))
    features, centers, nvoxels = getVoxelDescriptors(sm, buffer=1, version=2)
    reffeatures, refcenters, refnvoxels = np.load(
        os.path.join(curr_dir, "test_voxeldescriptors", "celecoxib_voxres.npy"),
        allow_pickle=True,
    )
    assert np.allclose(features, reffeatures)
    assert np.array_equal(centers, refcenters)
    assert np.array_equal(nvoxels, refnvoxels)


def _test_ledipasvir():
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

    sm = SmallMol(os.path.join(curr_dir, "test_voxeldescriptors", "ledipasvir.mol2"))
    features, centers, nvoxels = getVoxelDescriptors(sm, buffer=1, version=2)
    reffeatures, refcenters, refnvoxels = np.load(
        os.path.join(curr_dir, "test_voxeldescriptors", "ledipasvir_voxres.npy"),
        allow_pickle=True,
    )
    assert np.allclose(features, reffeatures)
    assert np.array_equal(centers, refcenters)
    assert np.array_equal(nvoxels, refnvoxels)


def _test_old_voxelization():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

    mol = Molecule(os.path.join(curr_dir, "test_voxeldescriptors", "3ptb.pdbqt"))
    mol.element[mol.element == "CA"] = "Ca"
    features, centers, nvoxels = getVoxelDescriptors(
        mol, buffer=8, voxelsize=1, version=1
    )
    reffeatures, refcenters, refnvoxels = np.load(
        os.path.join(curr_dir, "test_voxeldescriptors", "3PTB_voxres_old.npy"),
        allow_pickle=True,
    )
    assert np.allclose(features, reffeatures)
    assert np.array_equal(centers, refcenters)
    assert np.array_equal(nvoxels, refnvoxels)


@pytest.mark.skipif(not HAS_OPENBABEL, reason="Openbabel is not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Windows OBabel fails at atom typing"
)
def _test_featC(_mol):
    from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

    features, centers, nvoxels = getVoxelDescriptors(_mol, method="C", version=2)
    reffeatures, refcenters, refnvoxels = np.load(
        os.path.join(curr_dir, "test_voxeldescriptors", "3PTB_voxres.npy"),
        allow_pickle=True,
    )
    assert np.allclose(features, reffeatures)
    assert np.array_equal(centers, refcenters)
    assert np.array_equal(nvoxels, refnvoxels)


@pytest.mark.skipif(not HAS_OPENBABEL, reason="Openbabel is not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Windows OBabel fails at atom typing"
)
def _test_channels_with_metals():
    from moleculekit.molecule import Molecule, mol_equal
    from moleculekit.tools.voxeldescriptors import getChannels

    ref_channels = np.load(
        os.path.join(curr_dir, "test_voxeldescriptors", "1ATL_channels.npy")
    )

    ref_mol = Molecule(
        os.path.join(curr_dir, "test_voxeldescriptors", "1ATL_atomtyped.psf")
    )  # Need PSF to store the atomtypes!
    ref_mol.read(os.path.join(curr_dir, "test_voxeldescriptors", "1ATL_atomtyped.pdb"))

    mol = Molecule(
        os.path.join(curr_dir, "test_voxeldescriptors", "1ATL_prepared.psf")
    )  # Need PSF to store the correct charges!
    mol.read(os.path.join(curr_dir, "test_voxeldescriptors", "1ATL_prepared.pdb"))

    channels, mol_atomtyped = getChannels(mol)

    assert np.array_equal(ref_channels, channels)
    assert mol_equal(mol_atomtyped, ref_mol)
