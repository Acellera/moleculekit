# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.projection import Projection
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MetricSasa(Projection):
    """ Calculate solvent accessible surface area of a molecule.

    Implementation and documentation taken from MDtraj shrake_rupley code.

    Parameters
    ----------
    sel : str
        Atom selection string for atoms or residues for which to calculate the SASA.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    probeRadius : float
        The radius of the probe, in Angstrom.
    numSpherePoints : int
        The number of points representing the surface of each atom, higher values lead to more accuracy.
    mode : str
        In mode == 'atom', the extracted areas are resolved per-atom. In mode == 'residue', this is consolidated down
        to the per-residue SASA by summing over the atoms in each residue.

    Returns
    -------
    metr : MetricSasa object
    """
    def __init__(self, sel='protein', probeRadius=1.4, numSpherePoints=960, mode='atom'):
        super().__init__()

        self._probeRadius = probeRadius
        self._numSpherePoints = numSpherePoints
        self._mode = mode
        self._sel = sel

    def _calculateMolProp(self, mol, props='all'):
        props = ('radii', 'atom_mapping', 'sel') if props == 'all' else props
        res = {}

        sel = mol.atomselect(self._sel)
        if 'sel' in props:
            res['sel'] = sel

        if 'radii' in props:
            _ATOMIC_RADII = {'C': 1.5, 'F': 1.2, 'H': 0.4, 'N': 1.10, 'O': 1.05, 'S': 1.6, 'P': 1.6}
            elements = [n[0] for n in mol.name[sel]]
            atom_radii = np.vectorize(_ATOMIC_RADII.__getitem__)(elements)
            res['radii'] = np.array(atom_radii, np.float32) + self._probeRadius

        if 'atom_mapping' in props:
            if self._mode == 'atom':
                res['atom_mapping'] = np.arange(np.sum(sel), dtype=np.int32)
            elif self._mode == 'residue':
                from moleculekit.util import sequenceID
                res['atom_mapping'] = sequenceID((mol.resid[sel], mol.chain[sel], mol.segid[sel])).astype(np.int32)
            else:
                raise ValueError('mode must be one of "residue", "atom". "{}" supplied'.format(self._mode))

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
        radii = getMolProp('radii')
        atom_mapping = getMolProp('atom_mapping')
        sel = getMolProp('sel')

        xyz = np.swapaxes(np.swapaxes(np.atleast_3d(mol.coords[sel, :, :]), 1, 2), 0, 1)
        xyz = np.array(xyz.copy(), dtype=np.float32) / 10  # converting to nm

        try:
            from mdtraj.geometry._geometry import _sasa as sasa
        except ImportError:
            raise ImportError('To calculate SASA you need to install mdtraj with `conda install mdtraj -c omnia`')

        out = np.zeros((mol.numFrames, atom_mapping.max() + 1), dtype=np.float32)
        sasa(xyz, radii / 10, int(self._numSpherePoints), atom_mapping, out)  # Divide radii by 10 for nm
        return out

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
        atom_mapping = getMolProp('atom_mapping')
        atomsel = getMolProp('sel')

        if self._mode == 'atom':
            atomidx = np.where(atomsel)[0]
        elif self._mode == 'residue':
            _, firstidx = np.unique(atom_mapping, return_index=True)
            atomidx = np.where(atomsel)[0][firstidx]
        else:
            raise ValueError('mode must be one of "residue", "atom". "{}" supplied'.format(self._mode))

        from pandas import DataFrame
        types = []
        indexes = []
        description = []
        for i in atomidx:
            types += ['SASA']
            indexes += [i]
            description += ['SASA of {} {} {}'.format(mol.resname[i], mol.resid[i], mol.name[i])]
        return DataFrame({'type': types, 'atomIndexes': indexes, 'description': description})


import unittest
class _TestMetricSasa(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        from os import path
        
        mol = Molecule(path.join(home(dataDir='test-projections'), 'trajectory', 'filtered.pdb'))
        mol.read(path.join(home(dataDir='test-projections'), 'trajectory', 'traj.xtc'))
        mol.dropFrames(keep=[0, 1])  # Keep only two frames because it's super slow
        self.mol = mol

    def test_sasa_atom(self):
        from os import path
        from moleculekit.home import home

        metr = MetricSasa(mode='atom')
        sasaA = metr.project(self.mol.copy())
        sasaA_ref = np.load(path.join(home(dataDir='test-projections'), 'metricsasa', 'sasa_atom.npy'))
        assert np.allclose(sasaA, sasaA_ref, atol=7e-4)

    def test_sasa_residue(self):
        from os import path
        from moleculekit.home import home

        metr = MetricSasa(mode='residue')
        sasaR = metr.project(self.mol.copy())
        sasaR_ref = np.load(path.join(home(dataDir='test-projections'), 'metricsasa', 'sasa_residue.npy'))
        assert np.allclose(sasaR, sasaR_ref, atol=3e-3)


if __name__ == '__main__':
    unittest.main(verbosity=2)