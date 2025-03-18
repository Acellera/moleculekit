from moleculekit.molecule import Molecule
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_triclinic_wrapping():
    refdir = os.path.join(curr_dir, "dodecahedral_box")
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
