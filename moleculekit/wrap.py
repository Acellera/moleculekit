# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
import ctypes as ct
import numpy as np

from moleculekit.home import home


def wrap(coords, bonds, box, centersel=None):
    """
    Wrap the coords back into the unit cell.
    Molecules will remain continuous, so they may escape the bounds of the prinary unit cell.
    """

    import platform

    libdir = home(libDir=True)

    if coords.dtype != np.float32:
        raise ValueError("coords is not float32")

    if coords.ndim == 2:
        coords = coords[:, :, None]

    if coords.shape[1] != 3:
        raise RuntimeError("coords needs to be natoms x 3 x nframes")

    nframes = coords.shape[2]
    natoms = coords.shape[0]
    nbonds = bonds.shape[0]

    if coords.strides[0] != 12 * nframes or coords.strides[1] != 4 * nframes:
        # It's a view -- need to make a copy to ensure contiguity of memory
        coords = np.array(coords, dtype=np.float32)
    if coords.strides[0] != 12 * nframes or coords.strides[1] != 4 * nframes:
        raise ValueError("coords is a view with unsupported strides")

    if np.size(bonds, 1) != 2:
        raise RuntimeError("'bonds' not nbonds x 2 in length")
    if np.size(box, 0) != 3 or np.size(box, 1) != nframes:
        raise RuntimeError("'box' not 3 x nframes in length")

    if platform.system() == "Windows":
        ct.cdll.LoadLibrary(os.path.join(libdir, "libgcc_s_seh-1.dll"))
        if os.path.exists(os.path.join(libdir, "psprolib.dll")):
            ct.cdll.LoadLibrary(os.path.join(libdir, "psprolib.dll"))

    lib = ct.cdll.LoadLibrary(os.path.join(libdir, "libvmdparser.so"))

    c_nbonds = ct.c_int(nbonds)
    c_natoms = ct.c_int(natoms)
    c_nframes = ct.c_int(nframes)

    if centersel is None:
        centersel = np.array([-1], dtype=np.int32)
    centersel = np.append(centersel, np.array([-1], dtype=np.int32))
    c_centersel = centersel.ctypes.data_as(ct.POINTER(ct.c_int))

    c_coords = coords.ctypes.data_as(ct.POINTER(ct.c_float))
    c_bonds = (
        bonds.flatten().astype(np.int32).copy().ctypes.data_as(ct.POINTER(ct.c_int))
    )
    c_box = (
        box.T.flatten()
        .astype(np.float64)
        .copy()
        .ctypes.data_as(ct.POINTER(ct.c_double))
    )

    lib.wrap(c_bonds, c_coords, c_box, c_nbonds, c_natoms, c_nframes, c_centersel)

    return coords


from unittest import TestCase


class _TestWrap(TestCase):
    def test_wrap(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home

        datadir = os.path.join(home(dataDir="molecule-readers"), "multi-traj")
        mol = Molecule(os.path.join(datadir, "structure.pdb"))
        mol.read(os.path.join(datadir, "data", "e1s1_1", "output.xtc"))
        mol.wrap()

        boxsize = mol.coords.max(axis=0) - mol.coords.min(axis=0)

        assert np.abs(mol.box - boxsize).max() < 2


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
