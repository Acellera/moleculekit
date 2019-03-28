# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MetricRmsd(Projection):
    """ Calculates the RMSD of a set of trajectories to a reference structure

    Parameters
    ----------
    refmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        The reference Molecule to which we want to calculate the RMSD.
    trajrmsdstr : str
        Atom selection string for the trajectories from which to calculate the RMSD.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    trajalnstr : str, optional
        Atom selection string for the trajectories from which to align to the reference structure.
        If None, it defaults to the same as `trajrmsdstr`.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refrmsdstr : str, optional
        Atom selection string for the reference structure from which to calculate the RMSD. If None, it defaults to
        `trajrmsdstr`. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refalnstr : str, optional
        Atom selection string for the reference structure from which to align to the trajectories. If None, it defaults
        to `trajalnstr`. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    centerstr : str, optional
        Atom selection string around which to center the wrapping of the trajectories.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    pbc : bool, optional
        Enable or disable simulation wrapping.
    """
    def __init__(self, refmol, trajrmsdstr, trajalnstr=None, refrmsdstr=None, refalnstr=None, centerstr='protein', pbc=True):
        super().__init__()

        if trajalnstr is None:
            trajalnstr = trajrmsdstr
        if refalnstr is None:
            refalnstr = trajalnstr
        if refrmsdstr is None:
            refrmsdstr = trajrmsdstr

        self._refmol = refmol.copy()
        if self._refmol.numFrames > 1:
            logger.warning('Reference molecule contains multiple frames. MetricRmsd will calculate the RMSD to the frame set in the refmol.frame variable.')
            self._refmol.dropFrames(keep=self._refmol.frame)

        self._refalnsel = self._refmol.atomselect(refalnstr)
        self._refrmsdsel = self._refmol.atomselect(refrmsdstr)
        self._trajalnsel = trajalnstr
        self._trajrmsdsel = trajrmsdstr
        self._centersel = centerstr
        self._pbc = pbc

    def _calculateMolProp(self, mol, props='all'):
        props = ('trajalnsel', 'trajrmsdsel', 'centersel') if props == 'all' else props
        res = {}

        if 'trajalnsel' in props:
            res['trajalnsel'] = mol.atomselect(self._trajalnsel)
            if np.sum(res['trajalnsel']) == 0:
                raise NameError('Alignment selection resulted in 0 atoms.')
        if 'trajrmsdsel' in props:
            res['trajrmsdsel'] = mol.atomselect(self._trajrmsdsel)
            if np.sum(res['trajrmsdsel']) == 0:
                raise NameError('RMSD selection resulted in 0 atoms.')
        if 'centersel' in props and self._pbc:
            res['centersel'] = mol.atomselect(self._centersel)
            if np.sum(res['centersel']) == 0:
                raise NameError('Center selection resulted in 0 atoms.')

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
        from moleculekit.util import molRMSD

        mol = mol.copy()
        getMolProp = lambda prop: self._getMolProp(mol, prop)

        if self._pbc:
            mol.wrap(getMolProp('centersel'))
        #mol.coords = self._wrapPositions(mol.box, mol.coords, centersel)
        mol.align(sel=getMolProp('trajalnsel'), refmol=self._refmol, refsel=self._refalnsel)

        return molRMSD(mol, self._refmol, getMolProp('trajrmsdsel'), self._refrmsdsel)

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
        trajrmsdsel = self._getMolProp(mol, 'trajrmsdsel')
        from pandas import DataFrame
        types = ['rmsd']
        indexes = [np.where(trajrmsdsel)[0]]
        description = ['RMSD to reference structure.']
        return DataFrame({'type': types, 'atomIndexes': indexes, 'description': description})

    def _wrapPositions(self, box, pos, centersel):
        if box is None or np.sum(box) == 0:
            logger.warning('MetricRmsd: The given molecule does not contain box dimensions for wrapping.')
            return pos
        center = np.mean(pos[centersel, :, :], axis=0)
        origin = center - (box / 2)
        pos = pos - origin
        return np.mod(pos, box)


import unittest
class _TestMetricRMSD(unittest.TestCase):
    def test_metricrmsd(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        import numpy as np
        from os import path

        mol = Molecule(path.join(home(dataDir='test-projections'), 'trajectory', 'filtered.pdb'))
        mol.read(path.join(home(dataDir='test-projections'), 'trajectory', 'traj.xtc'))
        ref = mol.copy()

        ref.dropFrames(keep=0)
        mol.dropFrames(keep=np.arange(mol.numFrames-20, mol.numFrames)) # Keep only last 20 frames

        metr = MetricRmsd(ref, 'protein and name CA')
        data = metr.project(mol)

        lastrmsd = np.array([1.30797791,  1.29860222,  1.25042927,  1.31319737,  1.27044261,
                            1.40294552,  1.25354612,  1.30127883,  1.40618336,  1.18303752,
                            1.24414587,  1.34513164,  1.31932807,  1.34282494,  1.2261436 ,
                            1.36359048,  1.26243281,  1.21157813,  1.26476419,  1.29413617], dtype=np.float32)
        assert np.all(np.abs(data[-20:] - lastrmsd) < 0.001), 'RMSD calculation is broken'


if __name__ == "__main__":
    unittest.main(verbosity=2)