# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MetricCoordinate(Projection):
    """ Creates a MetricCoordinate object that calculates the atom coordinates from a set of trajectories.

    Parameters
    ----------
    atomsel : str
        Atom selection string for the atoms whose coordinates we want to calculate.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The reference Molecule to which we will align.
    trajalnsel : str, optional
        Atom selection string for the trajectories from which to align to the reference structure.
        If it's None and a `refmol` is passed it will default to 'protein and name CA'.
    refalnsel : str, optional
        Atom selection string for `refmol` from which to align to the reference structure. If None, it defaults to the
        same as `trajalnsel`.
    centersel : str, optional
        Atom selection string around which to wrap the simulation.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    pbc : bool
        Enable or disable coordinate wrapping based on periodic boundary conditions.

    Returns
    -------
    metr : MetricCoordinate object
    """
    def __init__(self, atomsel, refmol=None, trajalnsel=None, refalnsel=None, centersel='protein', pbc=True):
        super().__init__()

        if atomsel is None:
            raise ValueError('Atom selection cannot be None')
        
        self._trajalnsel = trajalnsel
        self._centersel = centersel
        self._atomsel = atomsel
        self._pbc = pbc

        self._refmol = refmol
        self._refalnsel = refalnsel
        if self._refmol is not None:
            if self._trajalnsel is None:
                self._trajalnsel = 'protein and name CA'
            self._refmol = refmol.copy()
            self._cache['refalnsel'] = self._refmol.atomselect(self._refalnsel if self._refalnsel is not None else self._trajalnsel)

    def _calculateMolProp(self, mol, props='all'):
        props = ('trajalnsel', 'atomsel', 'centersel') if props == 'all' else props
        res = {}
        if 'trajalnsel' in props:
            res['trajalnsel'] = None
            if self._trajalnsel is not None:
                res['trajalnsel'] = mol.atomselect(self._trajalnsel)
                if np.sum(res['trajalnsel']) == 0:
                    raise RuntimeError('Alignment selection resulted in 0 atoms.')                
        if 'atomsel' in props:
            res['atomsel'] = mol.atomselect(self._atomsel)
            if np.sum(res['atomsel']) == 0:
                raise RuntimeError('Atom selection resulted in 0 atoms.')
        if 'centersel' in props:
            res['centersel'] = mol.atomselect(self._centersel)
            if np.sum(res['centersel']) == 0:
                raise RuntimeError('Centering selection resulted in 0 atoms.')
        return res

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
            mol.wrap(getMolProp('centersel'))

        trajalnsel = getMolProp('trajalnsel')

        if trajalnsel is not None:
            if self._refmol is None:
                mol.align(trajalnsel)
            else:
                mol.align(trajalnsel, refmol=self._refmol, refsel=getMolProp('refalnsel'))

        coords = np.transpose(mol.coords[getMolProp('atomsel')], axes=(2, 1, 0)).reshape(mol.numFrames, -1)
        return coords

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

        atomidx = np.where(getMolProp('atomsel'))[0]
        from pandas import DataFrame
        types = []
        indexes = []
        description = []
        for xyz in ('X', 'Y', 'Z'):
            for i in atomidx:
                types += ['coordinate']
                indexes += [i]
                description += ['{} coordinate of {} {} {}'.format(xyz, mol.resname[i], mol.resid[i], mol.name[i])]
        return DataFrame({'type': types, 'atomIndexes': indexes, 'description': description})


import unittest
class _TestMetricCoordinate(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(path.join(home(dataDir='test-projections'), 'trajectory', 'filtered.pdb'))
        mol.read(path.join(home(dataDir='test-projections'), 'trajectory', 'traj.xtc'))

        self.mol = mol

    def test_project(self):
        ref = self.mol.copy()
        ref.coords = np.atleast_3d(ref.coords[:, :, 0])
        metr = MetricCoordinate('protein and name CA')
        data = metr.project(self.mol)

        lastcoors = np.array([-24.77000237, -27.76000023, -30.44000244, -33.65000153,
                              -33.40999985, -36.32000351, -36.02000427, -36.38000107,
                              -39.61000061, -41.01000214, -43.80000305, -45.56000137,
                              -45.36000061, -47.13000488, -49.54000473, -50.6000061 ,
                              -50.11000061, -52.15999985, -55.1400032 , -55.73000336], dtype=np.float32)
        assert np.all(np.abs(data[-1, -20:] - lastcoors) < 0.001), 'Coordinates calculation is broken'

    def test_project_align(self):
        ref = self.mol.copy()
        ref.coords = np.atleast_3d(ref.coords[:, :, 0])
        metr = MetricCoordinate('protein and name CA', ref)
        data = metr.project(self.mol)

        lastcoors = np.array([6.79283285, 5.55226946, 4.49387407, 2.94484425,
                              5.36937141, 3.18590879, 5.75874281, 5.48864174,
                              1.69625032, 1.58790839, 0.57877392, -2.66498065,
                              -3.70919156, -3.33702421, -5.38465405, -8.43286991,
                              -8.15859032, -7.85062265, -10.92551327, -13.70733166], dtype=np.float32)
        assert np.all(np.abs(data[-1, -20:] - lastcoors) < 0.001), 'Coordinates calculation is broken'


if __name__ == '__main__':
    unittest.main(verbosity=2)

