from moleculekit.molecule import Molecule
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_orthogonal_wrapping():
    mol = Molecule(os.path.join(curr_dir, "test_wrapping", "structure.prmtop"))
    mol.read(os.path.join(curr_dir, "test_wrapping", "output.xtc"))
    mol.wrap("protein or resname ACE NME")

    refmol = Molecule(os.path.join(curr_dir, "test_wrapping", "structure.prmtop"))
    refmol.read(os.path.join(curr_dir, "test_wrapping", "output_wrapped.xtc"))
    assert np.allclose(mol.coords, refmol.coords, atol=1e-2)

    # Test wrapping around a specific center coordinate
    mol = Molecule(os.path.join(curr_dir, "test_wrapping", "6X18.psf"))
    mol.read(os.path.join(curr_dir, "test_wrapping", "6X18.xtc"))
    newcenter = np.array([94.64, 3.69, 1.11])
    assert np.linalg.norm(mol.coords[:, :, 0].mean(axis=0) - newcenter) > 100
    mol.wrap(wrapcenter=newcenter)
    assert np.linalg.norm(mol.coords[:, :, 0].mean(axis=0) - newcenter) < 1


def _test_triclinic_wrapping():
    refdir = os.path.join(curr_dir, "test_readers", "dodecahedral_box")
    mol = Molecule(os.path.join(refdir, "3ptb_dodecahedron.psf"))

    mol.read(os.path.join(refdir, "output.xtc"))
    mol.wrap(wrapsel="protein", unitcell="triclinic")
    refcoords = Molecule(os.path.join(refdir, "output_triclinic_wrapped.xtc"))
    assert np.max(np.abs(mol.coords - refcoords.coords)) < 1e-2

    mol.read(os.path.join(refdir, "output.xtc"))
    mol.wrap(wrapsel="protein", unitcell="compact")
    refcoords = Molecule(os.path.join(refdir, "output_compact_wrapped.xtc"))
    assert np.max(np.abs(mol.coords - refcoords.coords)) < 1e-2

    mol.read(os.path.join(refdir, "output.xtc"))
    mol.wrap(wrapsel="protein", unitcell="rectangular")
    refcoords = Molecule(os.path.join(refdir, "output_rectangular_wrapped.xtc"))
    assert np.max(np.abs(mol.coords - refcoords.coords)) < 1e-2
