# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MetricShell(Projection):
    """ Calculates the density of atoms around other atoms.

    The MetricShell class calculates the density of a set of
    interchangeable atoms in concentric spherical shells around some
    other atoms. Thus it can treat identical molecules (like water or
    ions) and calculate summary values like the changes in water density
    around atoms. It produces a n-by-s dimensional vector where n the
    number of atoms in the first selection and s the number of shells
    around each of the n atoms.

    Parameters
    ----------
    sel1 : str
        Atom selection string for the first set of atoms around which the shells will be calculated.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    sel2 : str
        Atom selection string for the second set of atoms whose density will be calculated in shells around `sel1`.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    numshells : int, optional
        Number of shells to use around atoms of `sel1`
    shellwidth : int, optional
        The width of each concentric shell in Angstroms
    pbc : bool, optional
        Set to false to disable distance calculations using periodic distances
    gap : int, optional
        Not functional yet
    truncate : float, optional
        Set all distances larger than `truncate` to `truncate`
    """
    def __init__(self, sel1, sel2, numshells=4, shellwidth=3, pbc=True, gap=None, truncate=None):
        super().__init__()

        from moleculekit.projections.metricdistance import MetricDistance
        self.metricdistance = MetricDistance(sel1=sel1, sel2=sel2, groupsel1=None, groupsel2=None, metric='distances', threshold=8, pbc=pbc, truncate=truncate)

        self.numshells = numshells
        self.shellwidth = shellwidth
        self.description = None
        self.shellcenters = None

    def _calculateMolProp(self, mol, props='all'):
        props = ('shellcenters', 'map') if props == 'all' else props
        res = {}

        mapping = np.vstack(self.metricdistance.getMapping(mol).atomIndexes)
        if 'map' in props:
            res['map'] = mapping
        if 'shellcenters' in props:
            res['shellcenters'] = np.unique(mapping[:, 0])
        return res

    def project(self, mol):
        """ Project molecule.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>`
            A :class:`Molecule <moleculekit.molecule.Molecule>` object to project.
        kwargs :
            Do not use this argument. Only used for backward compatibility. Will be removed in later versions.

        Returns
        -------
        data : np.ndarray
            An array containing the projected data.
        """
        molprops = self._getMolProp(mol, 'all')
        distances = self.metricdistance.project(mol)
        if distances.ndim == 1:
            distances = distances[np.newaxis, :]
        return _shells(distances, molprops['map'][:, 0], molprops['shellcenters'], self.numshells, self.shellwidth)

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
        shellcenters = self.metricdistance._getMolProp(mol, 'sel1')

        from pandas import DataFrame
        types = []
        indexes = []
        description = []
        for i in np.where(shellcenters)[0]:
            for n in range(self.numshells):
                types += ['shell']
                indexes += [i]
                description += ['Density of sel2 atoms in shell {}-{} A centered on atom {} {} {}'
                                .format(n*self.shellwidth, (n+1)*self.shellwidth, mol.resname[i], mol.resid[i], mol.name[i])]
        return DataFrame({'type': types, 'atomIndexes': indexes, 'description': description})


def _shells(distances, map, shellcenters, numshells, shellwidth):
    shelledges = np.arange(shellwidth*(numshells+1), step=shellwidth)
    shellvol = 4/3 * np.pi * (shelledges[1:] ** 3 - shelledges[:-1] ** 3)

    shellmetric = np.ones((np.size(distances, 0), len(shellcenters) * numshells)) * -1

    for i in range(len(shellcenters)):
        cols = map == shellcenters[i]
        for e in range(len(shelledges)-1):
            inshell = (distances[:, cols] > shelledges[e]) & (distances[:, cols] <= shelledges[e+1])
            shellmetric[:, (i*numshells)+e] = np.sum(inshell, axis=1) / shellvol[e]

    return shellmetric


import unittest
class _TestMetricShell(unittest.TestCase):
    def test_metricshell(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path

        mol = Molecule(path.join(home(dataDir='test-projections'), 'trajectory', 'filtered.pdb'))
        mol.read(path.join(home(dataDir='test-projections'), 'trajectory', 'traj.xtc'))

        metr = MetricShell('protein and name CA', 'resname MOL and noh')
        data = metr.project(mol)

        densities = np.array([0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.00095589,  0.        ,  0.        ,  0.        ,  0.00023897,
                            0.        ,  0.        ,  0.        ,  0.00191177,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
        assert np.all(np.abs(data[193, 750:770] - densities) < 0.001), 'Shell density calculation is broken'


if __name__ == "__main__":
    unittest.main(verbosity=2)