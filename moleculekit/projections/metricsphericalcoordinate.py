# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MetricSphericalCoordinate(Projection):
    """ Creates a MetricSphericalCoordinate object that calculates the spherical coordinates between two centers of
    masses from a set of trajectories.

    Parameters
    ----------
    refmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The reference Molecule to which we will align.
    targetcom : str
        Atom selection string from which to calculate the target center of mass.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refcom : str
        Atom selection string from which to calculate the reference center of mass.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    trajalnsel : str, optional
        Atom selection string for the trajectories from which to align to the reference structure.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refalnsel : str, optional
        Atom selection string for `refmol` from which to align to the reference structure. If None, it defaults to the
        same as `trajalnstr`. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    centersel : str, optional
        Atom selection string around which to wrap the simulation.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    pbc : bool
        Enable or disable coordinate wrapping based on periodic boundary conditions.

    Returns
    -------
    metr : MetricCoordinate object
    """
    def __init__(self, refmol, targetcom, refcom, trajalnsel='protein and name CA', refalnsel=None, centersel='protein', pbc=True):
        super().__init__()

        if refalnsel is None:
            refalnsel = trajalnsel
        self._refmol = refmol
        self._refalnsel = self._refmol.atomselect(refalnsel)
        self._trajalnsel = trajalnsel
        self._centersel = centersel
        self._targetcom = targetcom
        self._refcom = refcom
        self._pbc = pbc

    def project(self, mol):
        """ Project molecule.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object to project.

        Returns
        -------
        data : np.ndarray
            An array containing the projected data.
        """
        getMolProp = lambda prop: self._getMolProp(mol, prop)

        mol = mol.copy()
        if self._pbc:
            mol.wrap(self._centersel)
        mol.align(getMolProp('trajalnsel'), refmol=self._refmol, refsel=self._refalnsel)

        refcom = np.mean(mol.coords[getMolProp('refcom'), :, :], axis=0)
        targetcom = np.mean(mol.coords[getMolProp('targetcom'), :, :], axis=0)
        xyz = targetcom - refcom

        r = np.sqrt(np.sum(xyz ** 2, axis=0))
        theta = np.arccos(xyz[2, :] / r)
        phi = np.arctan2(xyz[1, :], xyz[0, :])

        return np.stack((r, theta, phi), axis=1)

    def _calculateMolProp(self, mol, props='all'):
        props = ('trajalnsel', 'targetcom', 'refcom', 'centersel') if props == 'all' else props
        res = {}

        if 'trajalnsel' in props:
            res['trajalnsel'] = mol.atomselect(self._trajalnsel)
            if np.sum(res['trajalnsel']) == 0:
                raise NameError('Alignment selection resulted in 0 atoms.')
        if 'targetcom' in props:
            res['targetcom'] = mol.atomselect(self._targetcom)
            if np.sum(res['targetcom']) == 0:
                raise NameError('Atom selection for `targetcom` resulted in 0 atoms.')
        if 'refcom' in props:
            res['refcom'] = mol.atomselect(self._refcom)
            if np.sum(res['refcom']) == 0:
                raise NameError('Atom selection for `refcom` resulted in 0 atoms.')
        return res

    def getMapping(self, mol):
        """ Returns the description of each projected dimension.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            A Molecule object which will be used to calculate the descriptions of the projected dimensions.

        Returns
        -------
        map : :class:`DataFrame <pandas.core.frame.DataFrame>` object
            A DataFrame containing the descriptions of each dimension
        """
        getMolProp = lambda prop: self._getMolProp(mol, prop)

        targetatomidx = np.where(getMolProp('targetcom'))[0]
        refatomidx = np.where(getMolProp('refcom'))[0]
        from pandas import DataFrame
        types = ['r', 'theta', 'phi']
        indexes = [[targetatomidx, refatomidx]] * 3
        description = ['r', 'theta', 'phi']
        return DataFrame({'type': types, 'atomIndexes': indexes, 'description': description})


import unittest
class _TestMetricSphericalCoordinate(unittest.TestCase):
    def test_metricsphericalcoordinate(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(path.join(home(dataDir='test-projections'), 'trajectory', 'filtered.pdb'))
        ref = mol.copy()
        mol.read(path.join(home(dataDir='test-projections'), 'trajectory', 'traj.xtc'))

        res = MetricSphericalCoordinate(ref, 'resname MOL', 'within 8 of resid 98').project(mol)
        _ = MetricSphericalCoordinate(ref, 'resname MOL', 'within 8 of resid 98').getMapping(mol)

        ref_array = np.load(path.join(home(dataDir='test-projections'), 'metricsphericalcoordinate', 'res.npy'))
        assert np.allclose(res, ref_array, rtol=0, atol=1e-04)



if __name__ == "__main__":
    unittest.main(verbosity=2)

