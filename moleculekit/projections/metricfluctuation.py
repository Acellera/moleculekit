# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.projections.metriccoordinate import MetricCoordinate as _MetricCoordinate
from moleculekit.util import sequenceID
import numpy as np


class MetricFluctuation(_MetricCoordinate):
    """ Creates a MetricFluctuation object that calculates the squared fluctuation of atom positions in trajectories.

    Depending on the `refmol` option the projection either returns the fluctuation from the mean position of the atoms,
    or from coordinates of reference given in `refmol`. This means it calculates for atom coordinates (x,y,z) and
    reference coordinates (xr, yr, yz): (xr-x)**2+(yr-y)**2+(zr-z)**2. If `groupsel` is set to `residue` it will
    calculate the mean of this value over all atoms of the residue.
    To then get the RMSD you need to compute the square root (of the mean) of the results this projection returns over
    the desired atoms.

    Parameters
    ----------
    atomsel : str
        Atom selection string for the atoms whose fluctuations we want to calculate.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refmol : :class:`Molecule <moleculekit.molecule.Molecule>` object
        If `refmol` is None, MetricFluctuation will calculate the fluctuation of the atoms/residues around the trajectory mean.
        If a `refmol` is given, it will calculate the fluctuation around the reference atom positions after aligning.
    trajalnsel : str, optional
        Atom selection string for the trajectories from which to align to the reference structure.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    refalnsel : str, optional
        Atom selection string for `refmol` from which to align to the reference structure. If None, it defaults to the
        same as `trajalnsel`. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    centersel : str, optional
        Atom selection string around which to wrap the simulation.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    pbc : bool
        Enable or disable coordinate wrapping based on periodic boundary conditions.
    mode : str
        Set to 'atom' to get the fluctuation per atom. Set to 'residue' to get the mean fluctuation of the residue by
        grouping all of its atoms given in `atomsel`

    Returns
    -------
    metr : MetricFluctuation object

    Examples
    --------
    Calculate the fluctuation of atoms wrt their mean positions
    >>> MetricFluctuation('protein and name CA').project(mol)  
    Calculate the fluctuation of atoms wrt the reference structure
    >>> MetricFluctuation('protein and name CA', refmol).project(mol)
    Calculate the fluctuation of residues wrt their mean positions
    >>> MetricFluctuation('protein', mode='residue').project(mol)
    Calculate the fluctuation of residues wrt the reference structure
    >>> MetricFluctuation('protein', refmol, mode='residue').project(mol)
    """
    def __init__(self, atomsel, refmol=None, trajalnsel='protein and name CA', refalnsel=None, centersel='protein', pbc=True, mode='atom'):
        super().__init__(atomsel=atomsel, refmol=refmol, trajalnsel=trajalnsel, refalnsel=refalnsel, centersel=centersel, pbc=pbc)
        self._mode = mode

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
        coords = super().project(mol)

        if self._refmol is None:
            refcoords = np.mean(coords, axis=0)
        else:
            refcoords = _MetricCoordinate(atomsel=self._atomsel, refmol=self._refmol).project(self._refmol)

        mapping = super().getMapping(mol)
        xyzgroups = mapping.groupby('atomIndexes').groups
        numatoms = len(xyzgroups)

        resids = sequenceID(mol.resid)

        atomfluct = np.zeros((coords.shape[0], numatoms))
        squarediff = (coords - refcoords) ** 2
        atomresids = np.zeros(numatoms, dtype=int)
        for i, atom in enumerate(sorted(xyzgroups.values(), key=lambda x: x[0])):
            assert len(np.unique(mapping.atomIndexes[atom])) == 1
            atomfluct[:, i] = squarediff[:, atom].sum(axis=1)
            atomresids[i] = resids[int(mapping.atomIndexes[atom[0]])]

        if self._mode == 'atom':
            return atomfluct
        elif self._mode == 'residue':
            numres = len(np.unique(atomresids))
            meanresfluct = np.zeros((coords.shape[0], numres))
            for i, r in enumerate(np.unique(atomresids)):
                meanresfluct[:, i] = atomfluct[:, atomresids == r].mean(axis=1)
            return meanresfluct
        else:
            raise RuntimeError('Invalid mode {} given. Choose between `atom` and `residue`'.format(self._mode))
            

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
        if self._mode == 'atom':
            for i in atomidx:
                types += ['fluctuation']
                indexes += [i]
                description += ['Fluctuation of {} {} {}'.format(mol.resname[i], mol.resid[i], mol.name[i])]
        elif self._mode == 'residue':
            resids = mol.resid[atomidx]
            for r in np.unique(resids):
                types += ['fluctuation']
                i = atomidx[np.where(resids == r)[0][0]]
                indexes += [i]
                description += ['Mean fluctuation of {} {}'.format(mol.resname[i], mol.resid[i])]
        else:
            raise RuntimeError('Invalid mode {} given. Choose between `atom` and `residue`'.format(self._mode))

        return DataFrame({'type': types, 'atomIndexes': indexes, 'description': description})



import unittest
class _TestMetricFluctuation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.molecule import Molecule
        from moleculekit.home import home
        import numpy as np
        from os import path

        mol = Molecule(path.join(home(dataDir='test-projections'), 'trajectory', 'filtered.pdb'))
        mol.read(path.join(home(dataDir='test-projections'), 'trajectory', 'traj.xtc'))
        ref = mol.copy()

        ref.dropFrames(keep=0)
        mol.dropFrames(keep=np.arange(mol.numFrames-20, mol.numFrames)) # Keep only last 20 frames
        self.ref = ref
        self.mol = mol

    def test_metricfluctuation_atom_ref(self):
        import os
        from moleculekit.home import home

        metr = MetricFluctuation('protein and name CA', self.ref)
        _ = metr.getMapping(self.mol)  # Just test if it executes fine
        data = metr.project(self.mol)
        refdata = np.load(os.path.join(home(dataDir='test-projections'), 'metricfluctuation', 'fluctuation_atom_ref.npy'))
        assert np.allclose(data, refdata, atol=1e-3), 'Atom to ref fluctuation calculation is broken'

    def test_metricfluctuation_atom_mean(self):
        import os
        from moleculekit.home import home

        metr = MetricFluctuation('protein and name CA')
        _ = metr.getMapping(self.mol)  # Just test if it executes fine
        data = metr.project(self.mol)
        refdata = np.load(os.path.join(home(dataDir='test-projections'), 'metricfluctuation', 'fluctuation_atom_mean.npy'))
        assert np.allclose(data, refdata, atol=1e-3), 'Atom mean fluctuation calculation is broken'

    def test_metricfluctuation_residue_ref(self):
        import os
        from moleculekit.home import home

        metr = MetricFluctuation('protein and noh', self.ref, mode='residue')
        _ = metr.getMapping(self.mol)  # Just test if it executes fine
        data = metr.project(self.mol)
        refdata = np.load(os.path.join(home(dataDir='test-projections'), 'metricfluctuation', 'fluctuation_residue_ref.npy'))
        assert np.allclose(data, refdata, atol=1e-3), 'Residue to ref fluctuation calculation is broken'
    
    def test_metricfluctuation_residue_mean(self):
        import os
        from moleculekit.home import home

        metr = MetricFluctuation('protein and noh', mode='residue')
        _ = metr.getMapping(self.mol)  # Just test if it executes fine
        data = metr.project(self.mol)
        refdata = np.load(os.path.join(home(dataDir='test-projections'), 'metricfluctuation', 'fluctuation_residue_mean.npy'))
        assert np.allclose(data, refdata, atol=1e-3), 'Residue mean fluctuation calculation is broken'


if __name__ == '__main__':
    unittest.main(verbosity=2)



        

